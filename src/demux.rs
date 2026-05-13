//! AMV container demuxer.
//!
//! Layout (re-derived from real AMV files plus the chunk tree documented in
//! ffmpeg's `libavformat/amvenc.c`):
//!
//! ```text
//! RIFF <size> "AMV "
//!   LIST <size> "hdrl"
//!     "amvh" 56 [us_per_frame, 28 reserved, width, height, fps_den, fps_num,
//!                4 reserved, duration]   (all LE u32; "size" fields throughout
//!                                         the file are routinely zero)
//!     LIST <size> "strl"
//!       "strh" 56  (zeroed)
//!       "strf" 36  (zeroed)
//!     LIST <size> "strl"
//!       "strh" 48  (zeroed)
//!       "strf" 20  (audio WAVEFORMATEX: format=1, channels, sample_rate,
//!                                       byte_rate, block_align, bps=16, +pad)
//!   LIST <size> "movi"
//!     <fourcc 00dc | 01wb> <le32 payload size> <payload>   ← interleaved V/A
//!     ...
//!   "AMV_"
//!   "END_"
//! ```
//!
//! Conformance / leniency:
//!
//! * The "size" fields on top-level LIST/RIFF chunks are commonly zero
//!   ("some players break when they're set correctly" — ffmpeg amvenc
//!   comment). We don't trust them; we walk by tag and use the per-chunk
//!   size only on `00dc` / `01wb`.
//! * No 2-byte chunk padding between `00dc` / `01wb` (verified against
//!   real ffmpeg output — `vsubhash`-style AMV writes go straight on,
//!   regardless of payload parity).

use std::io::Read;

use oxideav_core::ContainerRegistry;
use oxideav_core::{
    CodecId, CodecParameters, CodecResolver, Error, MediaType, Packet, PixelFormat, Rational,
    Result, SampleFormat, StreamInfo, TimeBase,
};
use oxideav_core::{Demuxer, ProbeData, ReadSeek};

pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer("amv", open);
    reg.register_extension("amv", "amv");
    reg.register_probe("amv", probe);
}

/// `RIFF????AMV ` — RIFF with form type `AMV ` (trailing space).
fn probe(p: &ProbeData) -> u8 {
    if p.buf.len() >= 12 && &p.buf[0..4] == b"RIFF" && &p.buf[8..12] == b"AMV " {
        100
    } else if p.ext == Some("amv") {
        25
    } else {
        0
    }
}

fn open(mut input: Box<dyn ReadSeek>, _codecs: &dyn CodecResolver) -> Result<Box<dyn Demuxer>> {
    let mut blob = Vec::new();
    input.read_to_end(&mut blob)?;
    if blob.len() < 12 {
        return Err(Error::invalid("AMV: file shorter than RIFF/AMV header"));
    }
    if &blob[0..4] != b"RIFF" || &blob[8..12] != b"AMV " {
        return Err(Error::invalid("AMV: not a RIFF/AMV file"));
    }

    let header = parse_amvh(&blob)?;

    // Locate the start of the `movi` body (just past the `LIST<size>movi`
    // sequence). ffmpeg writes a fixed top-level layout; we walk the body of
    // the RIFF chunk looking for the next LIST whose form-type is `movi`.
    let movi_start = find_movi_start(&blob)?;

    let v_params = {
        let mut p = CodecParameters::video(CodecId::new(crate::VIDEO_CODEC_ID_STR));
        p.media_type = MediaType::Video;
        p.width = Some(header.width);
        p.height = Some(header.height);
        p.pixel_format = Some(PixelFormat::Yuv420P);
        if header.fps_den > 0 {
            p.frame_rate = Some(Rational::new(header.fps_num as i64, header.fps_den as i64));
        }
        p
    };
    let a_params = {
        let mut p = CodecParameters::audio(CodecId::new(crate::AUDIO_CODEC_ID_STR));
        p.media_type = MediaType::Audio;
        p.channels = Some(header.audio_channels.max(1));
        p.sample_rate = Some(header.audio_sample_rate);
        p.sample_format = Some(SampleFormat::S16);
        p
    };

    // Stream 0 = video, stream 1 = audio. AMV requires strict V-A
    // interleaving so this index assignment matches packet order.
    let v_tb = if header.fps_den > 0 && header.fps_num > 0 {
        TimeBase::new(header.fps_den as i64, header.fps_num as i64)
    } else {
        TimeBase::new(1, 30)
    };
    let a_tb = TimeBase::new(1, header.audio_sample_rate as i64);

    let streams = vec![
        StreamInfo {
            index: 0,
            time_base: v_tb,
            duration: None,
            start_time: Some(0),
            params: v_params,
        },
        StreamInfo {
            index: 1,
            time_base: a_tb,
            duration: None,
            start_time: Some(0),
            params: a_params,
        },
    ];

    Ok(Box::new(AmvDemuxer {
        blob,
        movi_start,
        cursor: movi_start,
        streams,
        v_pts: 0,
        a_pts: 0,
        eof: false,
        seek_index: Vec::new(),
        index_complete: false,
    }))
}

/// Locate the offset of the first byte inside the `movi` LIST body — i.e.
/// the byte right after the `"movi"` form-type.
fn find_movi_start(blob: &[u8]) -> Result<usize> {
    // Top-level RIFF body starts at offset 12.
    let mut i = 12usize;
    while i + 8 <= blob.len() {
        let tag = &blob[i..i + 4];
        let size =
            u32::from_le_bytes([blob[i + 4], blob[i + 5], blob[i + 6], blob[i + 7]]) as usize;
        i += 8;
        if tag == b"LIST" {
            if i + 4 > blob.len() {
                return Err(Error::invalid("AMV: truncated LIST"));
            }
            let form = &blob[i..i + 4];
            i += 4;
            if form == b"movi" {
                return Ok(i);
            }
            // Top-level sizes are commonly zero — fall back to walking the
            // children. For `hdrl` we don't actually need its body here so
            // skip via the size if non-zero, else jump to next RIFF tag by
            // walking child chunks.
            if size >= 4 && i + (size - 4) <= blob.len() {
                i += size - 4;
            } else {
                // size was bogus; walk children byte-by-byte until we find
                // the next plausible LIST tag.
                while i + 4 <= blob.len() && &blob[i..i + 4] != b"LIST" {
                    i += 1;
                }
            }
        } else {
            // Non-list top-level chunk (rare — amvh shouldn't be at top
            // level but accept it). Skip its declared size.
            if size > 0 && i + size <= blob.len() {
                i += size;
            } else {
                i += 1;
            }
        }
    }
    Err(Error::invalid("AMV: movi list not found"))
}

/// AMV main header (`amvh` chunk, 56 bytes).
#[derive(Clone, Copy, Debug, Default)]
struct AmvHeader {
    /// Width in pixels (must be >0).
    pub width: u32,
    /// Height in pixels (must be >0).
    pub height: u32,
    /// Frame-rate numerator (per ffmpeg's amvenc layout: stored at offset
    /// 44, paired with `fps_den` at 40).
    pub fps_num: u32,
    /// Frame-rate denominator.
    pub fps_den: u32,
    pub audio_sample_rate: u32,
    pub audio_channels: u16,
}

/// Walk the file looking for the `amvh` chunk inside `LIST hdrl`, plus the
/// audio `strf` (WAVEFORMATEX) inside the second `strl`. Robust to the
/// common case where every LIST/RIFF size field is zero.
fn parse_amvh(blob: &[u8]) -> Result<AmvHeader> {
    let mut h = AmvHeader::default();

    // Linear scan for known 4-CCs. The container is small and the tag set
    // is unambiguous (no risk of `amvh` colliding with payload bytes
    // because we stop scanning once we've seen `movi`).
    let mut i = 12usize;
    let mut saw_audio_strf = false;
    while i + 4 <= blob.len() {
        let tag = &blob[i..i + 4];
        if tag == b"movi" {
            break;
        }
        if tag == b"amvh" {
            if i + 8 + 56 > blob.len() {
                return Err(Error::invalid("AMV: amvh truncated"));
            }
            let body = &blob[i + 8..i + 8 + 56];
            // microseconds-per-frame at offset 0 — mostly redundant with
            // the explicit fps numerator/denominator further down, so we
            // ignore it for time-base computation.
            h.width = u32::from_le_bytes(body[32..36].try_into().unwrap());
            h.height = u32::from_le_bytes(body[36..40].try_into().unwrap());
            h.fps_den = u32::from_le_bytes(body[40..44].try_into().unwrap());
            h.fps_num = u32::from_le_bytes(body[44..48].try_into().unwrap());
            i += 8 + 56;
            continue;
        }
        if tag == b"strf" {
            // Audio strf is 20 bytes (WAVEFORMATEX). The first `strf`
            // encountered is the video one (size 36) which we ignore.
            let size = u32::from_le_bytes(blob[i + 4..i + 8].try_into().unwrap()) as usize;
            if !saw_audio_strf && size == 20 {
                if i + 8 + 20 > blob.len() {
                    return Err(Error::invalid("AMV: audio strf truncated"));
                }
                let body = &blob[i + 8..i + 8 + 20];
                let channels = u16::from_le_bytes(body[2..4].try_into().unwrap());
                let sample_rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                h.audio_channels = if channels == 0 { 1 } else { channels };
                h.audio_sample_rate = if sample_rate == 0 {
                    22_050
                } else {
                    sample_rate
                };
                saw_audio_strf = true;
            } else if size == 36 {
                // video strf — typically zeroed in AMV; ignore.
            }
            i += 8 + size;
            continue;
        }
        i += 1;
    }
    if h.width == 0 || h.height == 0 {
        return Err(Error::invalid("AMV: missing amvh / zero width/height"));
    }
    if h.audio_sample_rate == 0 {
        h.audio_sample_rate = 22_050;
        h.audio_channels = 1;
    }
    Ok(h)
}

/// One entry per video chunk discovered when (lazily) indexing the file.
///
/// We record the byte offset of the `00dc` chunk header along with the
/// per-stream PTS counters as they stood _just before_ that chunk was
/// emitted — i.e. seeking to this entry and resuming the walk reproduces
/// the same `(v_pts, a_pts)` the demuxer would have at this point in a
/// linear read. Audio chunks are not separately recorded: they live
/// strictly between consecutive video entries and are recovered by
/// resuming the walk after a seek.
#[derive(Clone, Copy, Debug)]
struct VideoIndexEntry {
    /// Byte offset of the `00dc` chunk header within `blob`.
    offset: usize,
    /// Video stream PTS for the frame at `offset` (frame counter).
    v_pts: i64,
    /// Cumulative audio PTS (samples) emitted before `offset` was reached.
    a_pts: i64,
}

struct AmvDemuxer {
    blob: Vec<u8>,
    /// First byte inside the `movi` LIST body.
    movi_start: usize,
    cursor: usize,
    streams: Vec<StreamInfo>,
    v_pts: i64,
    a_pts: i64,
    eof: bool,
    /// Lazy index of every `00dc` chunk in the file. Built incrementally
    /// the first time `seek_to` is called (or extended on subsequent
    /// seeks past the previously-scanned region). AMV has no built-in
    /// chunk index akin to AVI's `idx1`, so this is the canonical seek
    /// table — every entry is a keyframe (AMV video is intra-only).
    seek_index: Vec<VideoIndexEntry>,
    /// True once the index walk has reached the end of `movi` (a trailer
    /// marker, EOF, or a bogus oversized chunk). After this, the index
    /// is final and `last entry` corresponds to the final video frame.
    index_complete: bool,
}

impl Demuxer for AmvDemuxer {
    fn format_name(&self) -> &str {
        "amv"
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        if self.eof {
            return Err(Error::Eof);
        }
        loop {
            if self.cursor + 8 > self.blob.len() {
                self.eof = true;
                return Err(Error::Eof);
            }
            let tag = &self.blob[self.cursor..self.cursor + 4];
            // Trailer markers stop the walk.
            if tag == b"AMV_" || tag == b"END_" {
                self.eof = true;
                return Err(Error::Eof);
            }
            let size = u32::from_le_bytes(
                self.blob[self.cursor + 4..self.cursor + 8]
                    .try_into()
                    .unwrap(),
            ) as usize;
            // A bogus huge size means we've fallen off the end of the
            // movi list — stop instead of looping forever.
            if size > self.blob.len().saturating_sub(self.cursor + 8) {
                self.eof = true;
                return Err(Error::Eof);
            }
            let payload_start = self.cursor + 8;
            let payload_end = payload_start + size;
            self.cursor = payload_end;
            // No padding: AMV writes chunks back-to-back regardless of
            // payload parity (verified empirically against ffmpeg-produced
            // files; the AVI spec says pad-to-even but AMV ignores it).
            match tag {
                b"00dc" => {
                    let pts = self.v_pts;
                    self.v_pts += 1;
                    let data = self.blob[payload_start..payload_end].to_vec();
                    let mut pkt = Packet::new(0, self.streams[0].time_base, data);
                    pkt.pts = Some(pts);
                    pkt.dts = Some(pts);
                    pkt.duration = Some(1);
                    pkt.flags.keyframe = true;
                    return Ok(pkt);
                }
                b"01wb" => {
                    // Number of decoded samples per AMV audio chunk =
                    // (payload_size - 8) * 2 (8-byte header + packed
                    // nibbles, two samples per byte).
                    let n_samples = if size >= 8 {
                        ((size - 8) * 2) as i64
                    } else {
                        0
                    };
                    let pts = self.a_pts;
                    self.a_pts += n_samples;
                    let data = self.blob[payload_start..payload_end].to_vec();
                    let mut pkt = Packet::new(1, self.streams[1].time_base, data);
                    pkt.pts = Some(pts);
                    pkt.dts = Some(pts);
                    pkt.duration = Some(n_samples);
                    pkt.flags.keyframe = true;
                    return Ok(pkt);
                }
                _ => {
                    // Unknown / junk chunk — skip.
                    continue;
                }
            }
        }
    }

    fn duration_micros(&self) -> Option<i64> {
        // amvh stores duration as a packed HH:MM:SS triple at the tail of
        // the chunk; ffmpeg even acknowledges its own writer leaves it
        // zero on most files. Fall back to None and let callers compute
        // from the longest stream.
        None
    }

    /// Seek to the keyframe at or before `pts` in `stream_index`'s time
    /// base. AMV video is intra-only, so every video chunk is a
    /// keyframe; AMV audio is IMA-ADPCM with self-contained per-chunk
    /// state (initial predictor + step-index in the 8-byte header), so
    /// audio packets are likewise independently decodable. The seek
    /// always lands on a video chunk boundary — pulling audio after
    /// the seek will return the first `01wb` chunk that appears between
    /// the landed video chunk and the next, which is the AMV-correct
    /// pairing.
    fn seek_to(&mut self, stream_index: u32, pts: i64) -> Result<i64> {
        if (stream_index as usize) >= self.streams.len() {
            return Err(Error::invalid(format!(
                "AMV: stream index {stream_index} out of range"
            )));
        }
        // Make sure we've indexed enough of the file to either find a
        // hit or rule one out. For pts ≤ 0 the answer is always the
        // first entry (or movi_start itself if there are no video
        // chunks) so we still need at least one index pass to know the
        // first entry's pts.
        self.ensure_index_covers(stream_index, pts);

        if self.seek_index.is_empty() {
            // No video chunks in the file at all. Reset to the start
            // of `movi` and report pts=0 in the requested time base.
            self.cursor = self.movi_start;
            self.v_pts = 0;
            self.a_pts = 0;
            self.eof = false;
            return Ok(0);
        }

        // Pick the index entry to land on, depending on which stream
        // the caller is seeking in. We always physically land on a
        // video chunk; only the comparison key changes.
        let landed_idx: usize = match stream_index {
            0 => {
                // Video stream: index by v_pts (== frame counter).
                if pts <= self.seek_index[0].v_pts {
                    0
                } else {
                    // Largest entry with v_pts <= pts.
                    match self.seek_index.binary_search_by(|e| e.v_pts.cmp(&pts)) {
                        Ok(i) => i,
                        Err(i) => {
                            // `i` is the insertion point; the entry
                            // before it is the largest <= pts.
                            i.saturating_sub(1).min(self.seek_index.len() - 1)
                        }
                    }
                }
            }
            1 => {
                // Audio stream: index by a_pts (== cumulative samples
                // already emitted _before_ the indexed video chunk).
                // Find the largest video entry whose a_pts <= pts —
                // that's the V chunk we should land on so the next
                // `next_packet()` re-emits the matching A chunk.
                if pts <= self.seek_index[0].a_pts {
                    0
                } else {
                    match self.seek_index.binary_search_by(|e| e.a_pts.cmp(&pts)) {
                        Ok(i) => i,
                        Err(i) => i.saturating_sub(1).min(self.seek_index.len() - 1),
                    }
                }
            }
            _ => unreachable!("stream_index bounds already checked"),
        };

        let entry = self.seek_index[landed_idx];
        self.cursor = entry.offset;
        self.v_pts = entry.v_pts;
        self.a_pts = entry.a_pts;
        self.eof = false;

        // Report the landed pts _in the caller's stream time base_.
        Ok(match stream_index {
            0 => entry.v_pts,
            1 => entry.a_pts,
            _ => unreachable!(),
        })
    }
}

impl AmvDemuxer {
    /// Extend the lazy seek index until either (a) the requested
    /// `pts` is provably bracketed (the latest indexed entry has the
    /// stream's pts strictly greater than `pts`), or (b) we've reached
    /// the end of `movi` and the index is complete.
    ///
    /// `stream_index` selects which pts axis to compare against:
    /// stream 0 → `v_pts` (frame counter), stream 1 → `a_pts`
    /// (cumulative sample count). Index construction itself records
    /// both axes regardless; only the "have we scanned far enough"
    /// stopping condition uses the selected axis.
    fn ensure_index_covers(&mut self, stream_index: u32, pts: i64) {
        if self.index_complete {
            return;
        }
        // Walk byte-for-byte starting from where the index left off
        // (resume cursor = end of last indexed chunk's payload, or
        // movi_start if the index is empty).
        let mut walk = match self.seek_index.last().copied() {
            Some(last) => {
                // Re-derive the cursor by reading the chunk size at
                // last.offset and skipping past its payload. The
                // alternative (storing payload_end per entry) doubles
                // memory traffic for no real saving.
                let size_at = chunk_payload_size(&self.blob, last.offset).unwrap_or(0);
                last.offset + 8 + size_at
            }
            None => self.movi_start,
        };
        // Replay the v_pts / a_pts counters as they were _just after_
        // the last indexed chunk's payload was emitted (= just before
        // the next chunk). When the index is empty, both are zero
        // (the demuxer starts both counters at zero, mirroring open()).
        let (mut v_pts, mut a_pts) = match self.seek_index.last().copied() {
            // The last indexed entry's a_pts is the audio cumulative
            // _before_ that video chunk, so we must add the audio
            // emitted between it and the next video chunk during the
            // walk. Easier to reset both from the entry: v_pts after
            // emitting that video chunk is last.v_pts + 1, but any
            // audio chunks between it and the next video are unindexed
            // — we'll re-count them as we walk.
            Some(last) => (last.v_pts + 1, last.a_pts),
            None => (0, 0),
        };

        loop {
            if walk + 8 > self.blob.len() {
                self.index_complete = true;
                return;
            }
            let tag = &self.blob[walk..walk + 4];
            if tag == b"AMV_" || tag == b"END_" {
                self.index_complete = true;
                return;
            }
            let size =
                u32::from_le_bytes(self.blob[walk + 4..walk + 8].try_into().unwrap()) as usize;
            if size > self.blob.len().saturating_sub(walk + 8) {
                // Same defensive bail-out as next_packet() — the
                // index ends here.
                self.index_complete = true;
                return;
            }
            let payload_start = walk + 8;
            let payload_end = payload_start + size;
            match tag {
                b"00dc" => {
                    self.seek_index.push(VideoIndexEntry {
                        offset: walk,
                        v_pts,
                        a_pts,
                    });
                    v_pts += 1;
                    walk = payload_end;
                    // Early-exit: did we just bracket the caller's pts?
                    let bracket_pts = match stream_index {
                        0 => v_pts, // next frame's pts
                        1 => a_pts, // audio not advanced by 00dc
                        _ => unreachable!(),
                    };
                    if bracket_pts > pts {
                        return;
                    }
                }
                b"01wb" => {
                    let n_samples = if size >= 8 {
                        ((size - 8) * 2) as i64
                    } else {
                        0
                    };
                    a_pts += n_samples;
                    walk = payload_end;
                    // For an audio-stream seek, bail as soon as the
                    // _cumulative_ a_pts overshoots: the next 00dc
                    // we'd index already starts past the target.
                    if stream_index == 1 && a_pts > pts {
                        return;
                    }
                }
                _ => {
                    walk = payload_end;
                }
            }
        }
    }
}

/// Read the LE32 payload size out of an 8-byte AMV chunk header at
/// `offset`. Returns `None` when the header runs past EOF — caller
/// treats that as "stop indexing here".
fn chunk_payload_size(blob: &[u8], offset: usize) -> Option<usize> {
    if offset + 8 > blob.len() {
        return None;
    }
    Some(u32::from_le_bytes(blob[offset + 4..offset + 8].try_into().unwrap()) as usize)
}
