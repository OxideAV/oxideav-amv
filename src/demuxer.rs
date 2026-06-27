//! AMV demuxer — `oxideav_core::Demuxer` trait implementation.
//!
//! The walker:
//!
//! 1. Reads the fixed-length prelude (offsets 0 .. 0x13C) and feeds it
//!    to [`crate::parse::AmvPrelude::parse`].
//! 2. Builds two [`StreamInfo`] entries (video then audio) and
//!    initialises the next-chunk cursor at `movi_payload_start`.
//! 3. On each [`Demuxer::next_packet`] call, reads an 8-byte chunk
//!    header at the cursor, classifies the tag via
//!    [`crate::parse::ChunkKind`], reads exactly `size` payload bytes,
//!    and advances by `8 + size` bytes with **no padding** (§4).
//! 4. On any non-`00dc` / non-`01wb` tag — most importantly the
//!    `AMV_END_` ASCII trailer that bounds the file — returns
//!    [`Error::Eof`] for subsequent calls. The trailer literally
//!    occupies the 8 bytes a header would, so a `read_exact` of the
//!    next "chunk header" lands on it cleanly.

use std::io::{Read, Seek, SeekFrom};

use oxideav_core::{
    ChannelLayout, CodecId, CodecParameters, Demuxer, Error, MediaType, Packet, PixelFormat,
    Rational, ReadSeek, Result, SampleFormat, StreamInfo, TimeBase,
};

use crate::parse::{
    AmvHeader, AmvPrelude, AmvWaveFormat, ChunkHeader, ChunkKind, AMV_END_TRAILER, PRELUDE_MIN_LEN,
};
use crate::AUDIO_CODEC_ID;

/// Stream indices used by the demuxer. Trace §3a (video) and §3b
/// (audio) define the in-file order; we preserve it 1:1 so callers
/// can compare against the documentation directly.
const STREAM_INDEX_VIDEO: u32 = 0;
const STREAM_INDEX_AUDIO: u32 = 1;

/// Crate-local demuxer error. Surfaced through the standalone API and
/// converted into [`Error`] for the framework wiring (see
/// `crate::lib::From<AmvDemuxerError> for oxideav_core::Error`).
#[derive(Debug, Clone)]
pub enum AmvDemuxerError {
    /// Underlying reader returned an `io::Error` or a short read.
    Io(String),
    /// A required FOURCC / length field did not match the expected
    /// value, or a header parser rejected a body.
    InvalidData(String),
    /// Walker hit the natural end of the `movi` payload (typically
    /// the `AMV_END_` trailer).
    Eof,
}

impl core::fmt::Display for AmvDemuxerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(s) => write!(f, "amv I/O error: {s}"),
            Self::InvalidData(s) => write!(f, "amv invalid data: {s}"),
            Self::Eof => write!(f, "amv end of stream"),
        }
    }
}

impl std::error::Error for AmvDemuxerError {}

impl From<std::io::Error> for AmvDemuxerError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

/// One entry in the optional chunk-index cache built by
/// [`AmvDemuxer::build_chunk_index`]. Records the file offset and the
/// per-stream cumulative PTS values **immediately before** the chunk
/// at that offset is consumed — so the corresponding `next_packet`
/// call after a seek lands on this very chunk and emits a packet whose
/// `pts == video_pts_before` (video chunks) or
/// `pts == audio_pts_before` (audio chunks).
///
/// AMV files have no embedded index — §1 quirk #2 — so this is built
/// lazily by a single forward walk of the `movi` payload. Once built,
/// subsequent seeks are binary-search lookups rather than O(N) walks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkIndexEntry {
    /// Absolute file offset of the 8-byte chunk header.
    pub file_offset: u64,
    /// Which stream's payload follows the 8-byte header.
    pub kind: ChunkKind,
    /// Cumulative video PTS **before** consuming this chunk. For a
    /// video chunk this is the `pts` that `next_packet` will stamp on
    /// the emitted packet; for an audio chunk this is the running
    /// video-frame counter that the audio chunk is interleaved with.
    pub video_pts_before: i64,
    /// Cumulative audio PTS (running decoded-sample count) **before**
    /// consuming this chunk. For an audio chunk this is the `pts` that
    /// `next_packet` will stamp on the emitted packet.
    pub audio_pts_before: i64,
    /// Cumulative count of `01wb` audio **blocks** (chunks) emitted
    /// **before** consuming this chunk — the chunk-count companion to
    /// [`Self::audio_pts_before`]'s sample count. Lets a seek snap the
    /// demuxer's [`AmvDemuxer::audio_blocks_emitted`] counter to the
    /// indexed position without re-walking, preserving the §4
    /// 1:1-pairing cross-check after an indexed seek.
    pub audio_blocks_before: u64,
}

/// The pure-Rust AMV demuxer. Owns a seekable byte reader (anything
/// implementing [`ReadSeek`]) and walks the AMV chunk tree as
/// described in the workspace trace document.
pub struct AmvDemuxer {
    reader: Box<dyn ReadSeek>,
    streams: [StreamInfo; 2],
    header: AmvHeader,
    audio_format: AmvWaveFormat,
    /// File offset of the first leaf chunk inside `movi` — the byte
    /// **after** the `movi` FOURCC. Recorded so `seek_to` can rewind
    /// to the start of the payload when the target PTS is behind the
    /// current cursor (AMV carries no index — §1 quirk #2 — so a
    /// backwards seek is a linear walk from `movi_start`).
    movi_start: u64,
    /// Cursor pointing at the **next chunk header** inside the `movi`
    /// payload. Advances by `8 + size` after every packet.
    cursor: u64,
    /// Once a non-`00dc` / non-`01wb` tag (typically the `AMV_END_`
    /// trailer) is observed, every subsequent `next_packet` returns
    /// [`Error::Eof`].
    eof: bool,
    /// Monotonically increasing video-frame index, used as both
    /// `pts` and `dts` on the video stream's `1/fps` clock.
    next_video_pts: i64,
    /// Number of decoded audio samples emitted so far. Updated after
    /// each audio chunk by reading the 8-byte preamble's
    /// `decoded sample count` field (§4b).
    next_audio_pts: i64,
    /// Number of `01wb` audio **blocks** (chunks) the walker has
    /// emitted so far. Distinct from [`Self::next_audio_pts`] (which
    /// accumulates *decoded samples* from the §4b preambles): this is
    /// the chunk count, incremented by exactly one per `01wb` chunk and
    /// rewound / fast-forwarded with the cursor exactly like
    /// `next_audio_pts`, so it equals the audio-block count drained at
    /// any walk position. It is the §4 1:1-pairing companion to
    /// `next_video_pts` — after a clean trailer-bounded drain the two
    /// counts are equal ("1116 `00dc` then 1116 `01wb`, perfectly
    /// paired", §4). See [`AmvDemuxer::audio_blocks_emitted`] /
    /// [`AmvDemuxer::movi_interleave_balanced`].
    next_audio_blocks: u64,
    /// Optional cached index of every chunk inside `movi`. Populated
    /// lazily by [`AmvDemuxer::build_chunk_index`]; once populated,
    /// [`AmvDemuxer::seek_to`] switches from a linear walk to a binary
    /// search.
    chunk_index: Option<Vec<ChunkIndexEntry>>,
    /// Whether the walker drained the stream via a graceful truncation
    /// recovery (set on a short-read at any chunk boundary that was not
    /// the [`AMV_END_TRAILER`] literal). See
    /// [`AmvDemuxer::is_truncated`]. Distinguishes
    /// "device-power-cut before trailer landed" from "saw `AMV_END_`
    /// cleanly".
    truncated: bool,
    /// Absolute file offset at which the §4c [`AMV_END_TRAILER`] literal
    /// was observed, recorded once the walk terminates cleanly on the
    /// trailer (`next_packet` / `seek_to` reading the 8 trailer bytes at
    /// the current cursor). `None` until the trailer is seen and `None`
    /// for a stream that drained via truncation instead (the trailer
    /// never landed). See [`AmvDemuxer::trailer_offset`].
    trailer_offset: Option<u64>,
}

impl std::fmt::Debug for AmvDemuxer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmvDemuxer")
            .field("header", &self.header)
            .field("audio_format", &self.audio_format)
            .field("movi_start", &self.movi_start)
            .field("cursor", &self.cursor)
            .field("eof", &self.eof)
            .field("next_video_pts", &self.next_video_pts)
            .field("next_audio_pts", &self.next_audio_pts)
            .field("next_audio_blocks", &self.next_audio_blocks)
            .field(
                "chunk_index_len",
                &self.chunk_index.as_ref().map(|v| v.len()),
            )
            .field("truncated", &self.truncated)
            .finish()
    }
}

impl AmvDemuxer {
    /// Open an AMV stream from a seekable reader. Reads the prelude
    /// immediately, exposing [`AmvDemuxer::header`] and
    /// [`AmvDemuxer::audio_format`] without consuming any `movi`
    /// payload.
    pub fn open<R>(reader: R) -> std::result::Result<Self, AmvDemuxerError>
    where
        R: ReadSeek + 'static,
    {
        Self::open_with(reader, false)
    }

    /// Open an AMV stream in **strict mode** — same as [`Self::open`]
    /// but the prelude pass also runs the §2 + §3 sentinel checks that
    /// [`crate::AmvHeader::validate_sentinels`] and
    /// `AmvPrelude::parse_strict` document: `dwMicroSecPerFrame ==
    /// 1_000_000 / fps`, `flag_one == 1`, `reserved_30 == 0`, and the
    /// four §3 stream-header bodies are entirely zero per the trace
    /// observation. Returns [`AmvDemuxerError::InvalidData`] when any
    /// cross-check fails — useful for tooling that wants to reject
    /// garbled / non-device-profile inputs up-front rather than emit
    /// packets from a corrupt header.
    ///
    /// The default [`Self::open`] entrypoint stays permissive so the
    /// existing demuxer-open path keeps accepting any byte-shaped
    /// prelude that satisfies the §1-§4 FOURCC layout.
    pub fn open_strict<R>(reader: R) -> std::result::Result<Self, AmvDemuxerError>
    where
        R: ReadSeek + 'static,
    {
        Self::open_with(reader, true)
    }

    /// Shared implementation behind [`Self::open`] (permissive) and
    /// [`Self::open_strict`] (strict §2/§3 sentinel validation). The
    /// only divergence is which prelude-parser entrypoint is invoked;
    /// everything downstream is identical so a strict-opened demuxer
    /// walks the same `movi` payload + trailer logic.
    fn open_with<R>(reader: R, strict: bool) -> std::result::Result<Self, AmvDemuxerError>
    where
        R: ReadSeek + 'static,
    {
        let mut reader: Box<dyn ReadSeek> = Box::new(reader);
        reader.seek(SeekFrom::Start(0))?;
        let mut prelude = vec![0u8; PRELUDE_MIN_LEN];
        reader.read_exact(&mut prelude)?;
        let parsed = if strict {
            AmvPrelude::parse_strict(&prelude)?
        } else {
            AmvPrelude::parse(&prelude)?
        };

        let header = parsed.header;
        let audio_format = parsed.audio_format;
        let movi_start = parsed.movi_payload_start;

        // Video stream: 1/fps clock, the `amv_video` direct codec id,
        // width/height from amvh. Frame-rate exposed as a Rational so
        // re-muxers (or PTS-aware codec wrappers) can recover the exact
        // cadence without rounding through micros_per_frame.
        //
        // The id is `amv_video` (`VIDEO_DIRECT_CODEC_ID`), **not** the
        // generic `mjpeg` (`VIDEO_CODEC_ID`): a `00dc` payload is a
        // table-stripped baseline JPEG (§4a) that a generic `mjpeg`
        // decoder cannot consume, so a pipeline resolving this stream
        // must route to the in-crate direct decoder this crate registers
        // under `amv_video`. The reconstruct-then-mjpeg route remains
        // reachable via the free `reconstruct_jpeg` splice for callers
        // that want a conforming JFIF for a generic decoder.
        let mut video_params = CodecParameters::video(CodecId::new(crate::VIDEO_DIRECT_CODEC_ID));
        video_params.media_type = MediaType::Video;
        video_params.width = Some(header.width);
        video_params.height = Some(header.height);
        // AMV's video planes are 4:2:0 in practice (the standard
        // sub-VGA portable-player MJPEG profile). The actual chroma
        // sampling is fixed by the player's hard-coded tables and
        // is not derivable from the container — we surface a sensible
        // default so downstream consumers don't have to special-case
        // a `None` pixel format. Downstream codecs may override.
        video_params.pixel_format = Some(PixelFormat::Yuv420P);
        video_params.frame_rate = Some(Rational::new(header.fps as i64, 1));
        let video_time_base = TimeBase::new(1, header.fps as i64);
        let video_stream = StreamInfo {
            index: STREAM_INDEX_VIDEO,
            time_base: video_time_base,
            duration: Some(header.duration_micros() / micros_per_tick(video_time_base)),
            start_time: Some(0),
            params: video_params,
        };

        // Audio stream: 1/samples_per_sec clock, ADPCM-AMV placeholder
        // codec id, mono. The WAVEFORMATEX *declared* PCM in the
        // header but the actual payload is ADPCM (§3b note); we
        // declare the codec id accordingly. The header's
        // `nSamplesPerSec` value (22 050) **is** the decoded sample
        // rate so it's the right `time_base` denominator.
        let mut audio_params = CodecParameters::audio(CodecId::new(AUDIO_CODEC_ID));
        audio_params.media_type = MediaType::Audio;
        audio_params.sample_rate = Some(audio_format.samples_per_sec);
        audio_params.channels = Some(audio_format.channels);
        // The in-crate §4b AMV-IMA-ADPCM decode
        // (`decode_audio_packet` / `decode_audio_payload`) emits the
        // 16-bit signed mono PCM the §3b `WAVEFORMATEX`'s
        // `wBitsPerSample = 16` declares — surface that decoded sample
        // format and the mono speaker layout so a downstream consumer
        // sizing an output buffer doesn't have to special-case a `None`
        // (the on-disk nibble payload is *not* this format, but the
        // decoded stream these params describe is).
        audio_params.sample_format = Some(SampleFormat::S16);
        audio_params.channel_layout = Some(ChannelLayout::from_count(audio_format.channels));
        let audio_time_base = TimeBase::new(1, audio_format.samples_per_sec as i64);
        let audio_stream = StreamInfo {
            index: STREAM_INDEX_AUDIO,
            time_base: audio_time_base,
            duration: Some(header.duration_micros() / micros_per_tick(audio_time_base)),
            start_time: Some(0),
            params: audio_params,
        };

        reader.seek(SeekFrom::Start(movi_start))?;

        Ok(Self {
            reader,
            streams: [video_stream, audio_stream],
            header,
            audio_format,
            movi_start,
            cursor: movi_start,
            eof: false,
            next_video_pts: 0,
            next_audio_pts: 0,
            next_audio_blocks: 0,
            chunk_index: None,
            truncated: false,
            trailer_offset: None,
        })
    }

    /// Borrow the parsed `amvh` header (resolution, fps, packed
    /// duration). Available without consuming any packets.
    pub fn header(&self) -> &AmvHeader {
        &self.header
    }

    /// Borrow the parsed audio `WAVEFORMATEX` (sample rate, declared
    /// format tag, bits-per-sample).
    pub fn audio_format(&self) -> &AmvWaveFormat {
        &self.audio_format
    }

    /// Decode a video [`Packet`] emitted by this demuxer straight to an
    /// upright RGB [`crate::DecodedFrame`].
    ///
    /// This is the demux→pixels one-call convenience: it binds the
    /// demuxer's parsed §2 `amvh` geometry (the only place the AMV video
    /// resolution exists — the `00dc` bitstream carries no frame header)
    /// to the packet's raw `00dc` payload and runs the in-crate
    /// §4a baseline-JPEG decoder ([`crate::decode_frame_from_payload`]).
    ///
    /// Returns [`Error::invalid`] if `packet` is not a video packet
    /// (stream index `0`), or surfaces the decode error for a malformed
    /// payload (missing SOI/EOI bracket, internal markers, zero geometry).
    pub fn decode_video_packet(&self, packet: &Packet) -> Result<crate::DecodedFrame> {
        if packet.stream_index != STREAM_INDEX_VIDEO {
            return Err(Error::invalid(format!(
                "amv: decode_video_packet expects a video packet (stream {STREAM_INDEX_VIDEO}), \
                 got stream {}",
                packet.stream_index
            )));
        }
        crate::decode_frame_from_payload(&self.header, &packet.data).map_err(Error::from)
    }

    /// Decode an audio [`Packet`] emitted by this demuxer straight to
    /// 16-bit mono PCM samples.
    ///
    /// This is the demux→PCM one-call convenience and the audio mirror
    /// of [`decode_video_packet`](Self::decode_video_packet): it runs
    /// the in-crate §4b AMV-IMA-ADPCM decoder
    /// ([`crate::decode_audio_payload`]) over the packet's full `01wb`
    /// payload — the 8-byte §4b preamble (predictor seed + decoded
    /// sample count) plus the nibble-packed body — and returns the
    /// `decoded_sample_count` 16-bit samples the §3b `WAVEFORMATEX`
    /// declares. Each block is self-contained (predictor re-seeded from
    /// the preamble, step index reset to 0), so no demuxer-side state
    /// carries across calls.
    ///
    /// The samples are mono at [`audio_format().samples_per_sec`](AmvWaveFormat)
    /// (22 050 Hz on the observed device profile).
    ///
    /// Returns [`Error::invalid`] if `packet` is not an audio packet
    /// (stream index `1`), or surfaces the decode error if the payload
    /// is shorter than the 8-byte preamble.
    pub fn decode_audio_packet(&self, packet: &Packet) -> Result<Vec<i16>> {
        if packet.stream_index != STREAM_INDEX_AUDIO {
            return Err(Error::invalid(format!(
                "amv: decode_audio_packet expects an audio packet (stream {STREAM_INDEX_AUDIO}), \
                 got stream {}",
                packet.stream_index
            )));
        }
        crate::decode_audio_payload(&packet.data).map_err(Error::from)
    }

    /// Convenience: current `movi`-walk cursor (file offset of the
    /// next chunk header). Exposed for tests + audit; not part of the
    /// stable contract.
    #[cfg(test)]
    pub(crate) fn cursor(&self) -> u64 {
        self.cursor
    }

    /// Walk the `movi` payload once, recording every chunk's file
    /// offset and the per-stream PTS values that precede it into an
    /// in-memory index. After this call returns, [`seek_to`](Self::seek_to)
    /// switches from a linear walk to a binary-search lookup over the
    /// index, which makes repeated random-access seeks O(log N) instead
    /// of O(N) and avoids re-reading every chunk header from disk for
    /// each backwards seek.
    ///
    /// The walk does not allocate large buffers — video chunk bodies
    /// are skipped via `Seek`, audio chunks have only their 8-byte §4b
    /// preamble read to recover the running decoded-sample count.
    ///
    /// AMV files have no embedded index (§1 quirk #2), so this is the
    /// equivalent of synthesising one. Idempotent: calling it twice
    /// rebuilds from scratch. The current walker state (`cursor`,
    /// `next_video_pts`, `next_audio_pts`, `next_audio_blocks`, `eof`)
    /// is preserved across the call so it can be invoked mid-walk
    /// without disturbing the caller's position.
    pub fn build_chunk_index(&mut self) -> std::result::Result<(), AmvDemuxerError> {
        // Save existing walker state so callers can invoke this mid-walk.
        let saved_cursor = self.cursor;
        let saved_video_pts = self.next_video_pts;
        let saved_audio_pts = self.next_audio_pts;
        let saved_audio_blocks = self.next_audio_blocks;
        let saved_eof = self.eof;

        let mut index = Vec::new();
        let mut cursor = self.movi_start;
        let mut video_pts: i64 = 0;
        let mut audio_pts: i64 = 0;
        let mut audio_blocks: u64 = 0;

        loop {
            self.reader.seek(SeekFrom::Start(cursor))?;
            let mut header_bytes = [0u8; 8];
            if self.reader.read_exact(&mut header_bytes).is_err() {
                // Truncated file before the trailer — accept what we
                // have, the index is still useful.
                break;
            }
            if header_bytes == AMV_END_TRAILER {
                break;
            }
            let mut tag = [0u8; 4];
            tag.copy_from_slice(&header_bytes[0..4]);
            let size = u32::from_le_bytes([
                header_bytes[4],
                header_bytes[5],
                header_bytes[6],
                header_bytes[7],
            ]);
            let chunk = ChunkHeader { tag, size };
            let kind = chunk.kind();
            match kind {
                ChunkKind::Video => {
                    index.push(ChunkIndexEntry {
                        file_offset: cursor,
                        kind,
                        video_pts_before: video_pts,
                        audio_pts_before: audio_pts,
                        audio_blocks_before: audio_blocks,
                    });
                    cursor += chunk.advance_total();
                    video_pts += 1;
                }
                ChunkKind::Audio => {
                    // Read only the 8-byte preamble to learn the
                    // decoded-sample count. A short read here means
                    // the file was truncated mid-preamble — surface
                    // the same graceful-EOF semantics the
                    // `next_packet` walker uses by recording the
                    // entry without a sample-count contribution and
                    // breaking the walk.
                    let mut preamble = [0u8; 8];
                    let preamble_len = preamble.len().min(chunk.size as usize);
                    if preamble_len > 0
                        && self
                            .reader
                            .read_exact(&mut preamble[..preamble_len])
                            .is_err()
                    {
                        // Drop the partially-readable chunk; the
                        // index covers everything that did land.
                        break;
                    }
                    let block_samples = if preamble_len >= 8 {
                        u32::from_le_bytes([preamble[4], preamble[5], preamble[6], preamble[7]])
                            as i64
                    } else {
                        0
                    };
                    index.push(ChunkIndexEntry {
                        file_offset: cursor,
                        kind,
                        video_pts_before: video_pts,
                        audio_pts_before: audio_pts,
                        audio_blocks_before: audio_blocks,
                    });
                    cursor += chunk.advance_total();
                    audio_pts += block_samples;
                    audio_blocks += 1;
                }
                ChunkKind::Other(other) => {
                    // Out-of-spec tag mid-walk. Surface it so the caller
                    // can decide whether the index is salvageable.
                    return Err(AmvDemuxerError::InvalidData(format!(
                        "amv: unexpected chunk tag {:?} during build_chunk_index at offset {:#x}",
                        std::str::from_utf8(&other).unwrap_or("?"),
                        cursor
                    )));
                }
            }
        }

        self.chunk_index = Some(index);

        // Restore walker state.
        self.cursor = saved_cursor;
        self.next_video_pts = saved_video_pts;
        self.next_audio_pts = saved_audio_pts;
        self.next_audio_blocks = saved_audio_blocks;
        self.eof = saved_eof;
        self.reader.seek(SeekFrom::Start(saved_cursor))?;

        Ok(())
    }

    /// Borrow the cached chunk index built by
    /// [`build_chunk_index`](Self::build_chunk_index). Returns `None`
    /// when the index has not been built yet.
    pub fn chunk_index(&self) -> Option<&[ChunkIndexEntry]> {
        self.chunk_index.as_deref()
    }

    /// Whether the walker drained the `movi` payload by hitting a
    /// truncation (a short read at a chunk boundary, or a short read
    /// inside a chunk body) instead of by observing the §4c
    /// [`AMV_END_TRAILER`] literal.
    ///
    /// Cheap portable-player devices that AMV originated on are prone
    /// to power-cut mid-write — battery dies, the user yanks the
    /// memory card — leaving a file whose last few bytes are missing
    /// the trailer (and possibly the last chunk's body). The walker
    /// tolerates these cases by returning [`Error::Eof`] for the
    /// truncating call, and this flag lets callers tell apart "I
    /// drained 1116 of 1116 frames cleanly" from "I drained 1043 of
    /// 1116 frames, the rest were lost when the device died".
    ///
    /// Always `false` before EOF is reached.
    pub fn is_truncated(&self) -> bool {
        self.truncated
    }

    /// Number of stream-0 video frames the walker has emitted so far.
    ///
    /// AMV interleaves chunks in a strict 1:1 video-first cadence (§4)
    /// and stamps each video chunk with a monotonically increasing
    /// `pts`/`dts` on the `1/fps` clock; the counter is incremented by
    /// exactly one per emitted `00dc` chunk, so after a complete drain
    /// it equals the total video-frame count the §4 `movi` walk found.
    /// For `comedian.amv` that is `1116` — the value the trace's §2
    /// worked example divides by `fps` to recover the packed duration
    /// (`1116 ÷ 12 = 93 s = 1:33`).
    ///
    /// Because AMV carries no `idx1` index and zeroes every RIFF / LIST
    /// size (§1 quirks), the only way to learn the frame count is to
    /// walk the payload — this accessor surfaces the count the demuxer
    /// already tracks for PTS stamping so a caller does not have to keep
    /// its own tally alongside the walk. The count reflects whatever has
    /// been drained at the moment it is read: it is `0` immediately after
    /// [`Self::open`], grows as [`Demuxer::next_packet`] emits video
    /// chunks, and on a truncated file (see [`Self::is_truncated`])
    /// reports the number of complete video chunks that landed before the
    /// device cut off. After a [`Demuxer::seek_to`] it reflects the seek
    /// target's frame index (the counter is rewound / fast-forwarded with
    /// the cursor), so read it only after a full forward drain when using
    /// it as a total.
    pub fn video_frames_emitted(&self) -> u64 {
        // `next_video_pts` is the PTS the *next* video chunk will carry;
        // since stamping starts at 0 and steps by 1 per emitted video
        // chunk, it equals the count already emitted.
        self.next_video_pts.max(0) as u64
    }

    /// Number of stream-1 `01wb` audio **blocks** (chunks) the walker
    /// has emitted so far — the chunk-count companion to
    /// [`Self::video_frames_emitted`].
    ///
    /// This is **not** the decoded-sample PTS (that is what the audio
    /// stream's `pts` tracks, accumulated from each block's §4b
    /// preamble); it is the count of `01wb` leaf chunks, incremented by
    /// exactly one per emitted audio packet. Under §4's strict 1:1
    /// video-first interleave every `00dc` is followed by exactly one
    /// `01wb`, so after a complete trailer-bounded drain this equals
    /// [`Self::video_frames_emitted`] — "1116 `00dc` then 1116 `01wb`,
    /// perfectly paired" (§4). For `comedian.amv` both reach `1116`,
    /// for `noel-son-lumiere.amv` both reach `2928`.
    ///
    /// Like the video counter, it reflects whatever has been drained at
    /// the moment it is read: `0` immediately after [`Self::open`], it
    /// grows as [`Demuxer::next_packet`] emits audio chunks, and it is
    /// rewound / fast-forwarded with the cursor by [`Demuxer::seek_to`]
    /// (including the indexed seek path, which snaps it to the target
    /// entry's [`ChunkIndexEntry::audio_blocks_before`]). On a truncated
    /// file (see [`Self::is_truncated`]) it reports the number of
    /// complete audio blocks that landed before the device cut off — so
    /// `video_frames_emitted() - audio_blocks_emitted()` is the count of
    /// video frames left unpaired by the power-cut. Use it as a stream
    /// total only after a full forward drain.
    pub fn audio_blocks_emitted(&self) -> u64 {
        self.next_audio_blocks
    }

    /// Confirm the §4 strict 1:1 video-first interleave came out
    /// **balanced** — exactly as many `01wb` audio blocks as `00dc`
    /// video frames — at the current walk position.
    ///
    /// Per trace §4 every observed `.amv` file pairs each video frame
    /// with exactly one audio block ("1116 `00dc` then 1116 `01wb`,
    /// perfectly paired"; the §5 hierarchy summary annotates the `movi`
    /// payload "strict 1:1, video-first"). Because the cadence is video
    /// first, during a walk the video count leads the audio count by at
    /// most one (right after a `00dc` is emitted but before its paired
    /// `01wb`); the two are equal exactly when the walk sits on a clean
    /// video/audio pair boundary.
    ///
    /// Returns `true` when [`Self::video_frames_emitted`] equals
    /// [`Self::audio_blocks_emitted`]. Read **after a complete forward
    /// drain to a clean trailer-bounded EOF** (see
    /// [`Self::trailer_offset`]), this is the demuxer-level, streaming
    /// equivalent of [`validate_movi_interleave`](crate::validate_movi_interleave)'s
    /// even-count rule — it confirms the §4 pairing held over the whole
    /// payload **without** buffering the entire chunk-kind sequence into
    /// a `Vec`, which the free function requires. On a truncated file
    /// (see [`Self::is_truncated`]) it typically returns `false`: the
    /// device cut off after a `00dc` but before its paired `01wb`,
    /// leaving the video count one ahead — exactly the unpaired-trailing
    /// -video signal the free function reports as a missing trailing
    /// audio block.
    ///
    /// Mid-walk it is `true` only at pair boundaries (after each `01wb`,
    /// before the next `00dc`), so a caller using it as a whole-stream
    /// integrity check must first drain to EOF.
    pub fn movi_interleave_balanced(&self) -> bool {
        self.video_frames_emitted() == self.audio_blocks_emitted()
    }

    /// Cross-check the §2 `amvh` packed-byte duration against the §4
    /// `movi`-walk video-frame count, applying the trace's central
    /// header↔payload invariant: "1116 frames ÷ 12 fps = 93 s = 1:33"
    /// (§2). Returns `true` when the header's packed
    /// `[seconds, minutes, hours]` triple equals
    /// [`AmvDuration::from_frame_count`](crate::AmvDuration::from_frame_count) computed from
    /// [`Self::video_frames_emitted`] and the header `fps`.
    ///
    /// This is the demuxer-level companion to
    /// [`AmvDuration::is_consistent_with_frame_count`](crate::AmvDuration::is_consistent_with_frame_count): that free
    /// function takes a caller-supplied frame count, whereas this method
    /// feeds it the count the walker actually drained, so a caller that
    /// has done a full forward drain can confirm in one call that the
    /// device-written duration agrees with the chunks that landed.
    ///
    /// Read this **only after a complete forward drain to EOF**. On a
    /// partial walk or right after a backwards [`Demuxer::seek_to`],
    /// [`Self::video_frames_emitted`] reflects the cursor position rather
    /// than the stream total, so the comparison would be against an
    /// incomplete count.
    ///
    /// For a **clean** trailer-bounded drain ([`Self::trailer_offset`]
    /// is `Some`) of a well-formed file this returns `true`. A truncated
    /// file (see [`Self::is_truncated`]) typically returns `false`: the
    /// header still records the device's intended full duration while the
    /// walk only reached the chunks that survived the power-cut — that
    /// disagreement is exactly the signal a recovery tool wants in order
    /// to decide whether to re-stamp the header via
    /// [`AmvDuration::from_frame_count`](crate::AmvDuration::from_frame_count).
    pub fn duration_consistent_with_drained_frames(&self) -> bool {
        self.header
            .duration()
            .is_consistent_with_frame_count(self.video_frames_emitted(), self.header.fps)
    }

    /// Absolute file offset at which the §4c [`AMV_END_TRAILER`] literal
    /// (`AMV_END_`) was observed, or `None` if the walk has not yet
    /// reached a clean trailer-bounded EOF.
    ///
    /// AMV files carry no `idx1` index and all RIFF / LIST sizes are
    /// zeroed (§1 quirks), so the 8-byte `AMV_END_` ASCII trailer is the
    /// *only* thing that bounds the `movi` payload (§4c "This trailer …
    /// is what bounds the stream for the device"). The trace's §4
    /// no-padding proof turns on exactly this offset: walking
    /// `comedian.amv` by `8 + size` per chunk for all 2232 chunks lands
    /// the cursor at `0x348E31`, which is where the `AMV_END_` trailer
    /// sits — `8` bytes before the `0x348E39` EOF. A demuxer that had
    /// mis-applied AVI even-byte padding would desync and never see the
    /// literal at a clean 8-byte boundary.
    ///
    /// The field is populated only when the walk terminates by reading
    /// the trailer literal at the chunk-header cursor — via
    /// [`Demuxer::next_packet`](oxideav_core::Demuxer::next_packet) or
    /// [`Demuxer::seek_to`](oxideav_core::Demuxer::seek_to). It stays
    /// `None` for a stream drained by truncation (the trailer never
    /// landed — see [`Self::is_truncated`]), so the two accessors
    /// together classify the three terminal states:
    ///
    /// * still walking → `trailer_offset() == None && !is_truncated()`,
    /// * clean trailer EOF → `trailer_offset() == Some(off)`,
    /// * truncated EOF → `trailer_offset() == None && is_truncated()`.
    ///
    /// On a clean walk the returned offset equals `file_len - 8` (the
    /// trailer is the final 8 bytes), so a caller that also knows the
    /// stream length can cross-check the no-padding walk arrived exactly
    /// where §4 says it should. See [`Self::trailer_matches_eof`].
    pub fn trailer_offset(&self) -> Option<u64> {
        self.trailer_offset
    }

    /// Cross-check the observed §4c trailer offset against a known total
    /// stream byte length, applying the §4 no-padding worked-example
    /// invariant: a clean `8 + size` walk lands the `AMV_END_` trailer
    /// at exactly `stream_len - 8` (the trailer is the final 8 bytes of
    /// the file, with no `idx1` index between the payload and EOF — §4c).
    ///
    /// Returns:
    ///
    /// * `Some(true)` when the trailer was observed
    ///   ([`Self::trailer_offset`] is `Some`) and its offset equals
    ///   `stream_len.checked_sub(8)` — for `comedian.amv` this is
    ///   `0x348E39 - 8 == 0x348E31`, the trace's recorded trailer
    ///   position.
    /// * `Some(false)` when the trailer was observed but its offset does
    ///   **not** match `stream_len - 8` (e.g. trailing garbage after the
    ///   trailer, or a `stream_len` that disagrees with the walk).
    /// * `None` when no trailer has been observed yet — the walk has not
    ///   reached a clean trailer-bounded EOF, so there is nothing to
    ///   cross-check (the caller should consult [`Self::is_truncated`]).
    ///
    /// `stream_len` is the caller-supplied total file length in bytes
    /// (e.g. from `std::fs::metadata`); the demuxer does not measure it
    /// itself because the [`ReadSeek`](oxideav_core::ReadSeek) contract
    /// is forward-walk oriented and a final length probe would perturb
    /// the cursor.
    pub fn trailer_matches_eof(&self, stream_len: u64) -> Option<bool> {
        self.trailer_offset
            .map(|off| stream_len.checked_sub(AMV_END_TRAILER.len() as u64) == Some(off))
    }
}

impl Demuxer for AmvDemuxer {
    fn format_name(&self) -> &str {
        crate::CONTAINER_NAME
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn next_packet(&mut self) -> Result<Packet> {
        if self.eof {
            return Err(Error::Eof);
        }
        // Read the next 8 bytes — either a chunk header or the
        // `AMV_END_` trailer (which is the same width and immediately
        // follows the last chunk per §4c).
        self.reader
            .seek(SeekFrom::Start(self.cursor))
            .map_err(|e| Error::other(format!("amv: seek to chunk header: {e}")))?;
        let mut header_bytes = [0u8; 8];
        if self.reader.read_exact(&mut header_bytes).is_err() {
            // A short read at this point means the file ended before
            // the trailer landed. AMV files routinely arrive truncated
            // from power-cut portable players — treat the truncation
            // as a graceful EOF and flag it so callers can distinguish
            // "saw trailer" from "stream cut off" via
            // [`AmvDemuxer::is_truncated`].
            self.eof = true;
            self.truncated = true;
            return Err(Error::Eof);
        }
        // First check the trailer literal — its 8 bytes match no
        // valid chunk-header layout (the would-be size field is
        // ASCII `b"END_"` = 0x5F444E45, which is enormous and not
        // sane; checking the literal first is simpler).
        if header_bytes == AMV_END_TRAILER {
            self.eof = true;
            // Record where the §4c trailer landed — the only byte-bound
            // the stream carries (§1 quirks: zeroed sizes, no idx1).
            self.trailer_offset = Some(self.cursor);
            return Err(Error::Eof);
        }

        let mut tag = [0u8; 4];
        tag.copy_from_slice(&header_bytes[0..4]);
        let size = u32::from_le_bytes([
            header_bytes[4],
            header_bytes[5],
            header_bytes[6],
            header_bytes[7],
        ]);
        let chunk = ChunkHeader { tag, size };
        match chunk.kind() {
            ChunkKind::Video => self.read_video_packet(chunk),
            ChunkKind::Audio => self.read_audio_packet(chunk),
            ChunkKind::Other(other) => {
                // Anything not in {00dc, 01wb, AMV_END_} is
                // out-of-spec per §4. Surface it as InvalidData so
                // the caller can decide whether to recover.
                Err(Error::invalid(format!(
                    "amv: unexpected chunk tag {:?} at offset {:#x}",
                    std::str::from_utf8(&other).unwrap_or("?"),
                    self.cursor
                )))
            }
        }
    }

    fn duration_micros(&self) -> Option<i64> {
        Some(self.header.duration_micros())
    }

    /// Seek to (or before) the requested presentation timestamp on the
    /// given stream.
    ///
    /// AMV carries no index — §1 quirk #2 ("there is no `idx1` index and
    /// no OpenDML `indx`/`ix##` meta-index") — so this is a **linear
    /// walk** over the `movi` payload. Header bytes are read but chunk
    /// bodies are skipped via [`Seek`] so the seek does not allocate
    /// large buffers for the JPEG video payloads on the way past.
    ///
    /// Every video chunk is a keyframe (`§4a` "intra-only"), so the
    /// stream-0 seek lands exactly at the chunk whose
    /// `pts == requested_pts` (or the last chunk if the request is
    /// beyond the end). The audio stream's PTS counter is the running
    /// decoded-sample count from §4b preambles, so a stream-1 seek
    /// lands at the chunk whose cumulative sample count first reaches
    /// or exceeds `requested_pts`.
    ///
    /// On success, the returned value is the PTS of the chunk the
    /// next [`next_packet`](Self::next_packet) call will emit on the
    /// requested stream. Returns [`Error::invalid`] when
    /// `requested_pts` is negative or `stream_index` is out of range.
    fn seek_to(&mut self, stream_index: u32, requested_pts: i64) -> Result<i64> {
        if requested_pts < 0 {
            return Err(Error::invalid(format!(
                "amv: requested_pts must be non-negative, got {requested_pts}"
            )));
        }
        if stream_index != STREAM_INDEX_VIDEO && stream_index != STREAM_INDEX_AUDIO {
            return Err(Error::invalid(format!(
                "amv: stream_index must be 0 (video) or 1 (audio), got {stream_index}"
            )));
        }

        // Fast path: if the chunk index has been built, binary-search
        // to find the first chunk on the requested stream whose
        // pre-emit PTS is >= requested_pts. This is O(log N) and skips
        // re-reading every chunk header from disk.
        if self.chunk_index.is_some() {
            return self.seek_to_via_index(stream_index, requested_pts);
        }

        // For stream 0 (video), `pts == frame index`. For stream 1
        // (audio), `pts == cumulative decoded-sample count`. A request
        // at or before the current running PTS rewinds to the start of
        // `movi` and walks forward; a request ahead of the current PTS
        // walks forward from the existing cursor.
        let current_pts = match stream_index {
            STREAM_INDEX_VIDEO => self.next_video_pts,
            STREAM_INDEX_AUDIO => self.next_audio_pts,
            _ => unreachable!(),
        };
        if requested_pts < current_pts {
            self.cursor = self.movi_start;
            self.eof = false;
            self.next_video_pts = 0;
            self.next_audio_pts = 0;
            self.next_audio_blocks = 0;
        }

        // Linear walk: read 8-byte chunk headers, advance past bodies
        // via Seek, track per-stream PTS, stop just before a chunk on
        // the requested stream whose next emitted PTS would equal or
        // exceed `requested_pts`. The chunk header peeked at this stop
        // point is not consumed — `next_packet` re-reads it via the
        // same `self.cursor` seek path.
        loop {
            self.reader
                .seek(SeekFrom::Start(self.cursor))
                .map_err(|e| Error::other(format!("amv: seek_to chunk header: {e}")))?;
            let mut header_bytes = [0u8; 8];
            if self.reader.read_exact(&mut header_bytes).is_err() {
                // Truncated file before we found the target — land on
                // EOF.
                self.eof = true;
                return Ok(match stream_index {
                    STREAM_INDEX_VIDEO => self.next_video_pts,
                    STREAM_INDEX_AUDIO => self.next_audio_pts,
                    _ => unreachable!(),
                });
            }
            if header_bytes == AMV_END_TRAILER {
                self.eof = true;
                self.trailer_offset = Some(self.cursor);
                return Ok(match stream_index {
                    STREAM_INDEX_VIDEO => self.next_video_pts,
                    STREAM_INDEX_AUDIO => self.next_audio_pts,
                    _ => unreachable!(),
                });
            }
            let mut tag = [0u8; 4];
            tag.copy_from_slice(&header_bytes[0..4]);
            let size = u32::from_le_bytes([
                header_bytes[4],
                header_bytes[5],
                header_bytes[6],
                header_bytes[7],
            ]);
            let chunk = ChunkHeader { tag, size };
            match chunk.kind() {
                ChunkKind::Video => {
                    if stream_index == STREAM_INDEX_VIDEO && self.next_video_pts >= requested_pts {
                        // Cursor still points at this chunk header —
                        // `next_packet` will emit it on the next call.
                        return Ok(self.next_video_pts);
                    }
                    // Skip the body — exactly `size` bytes, no padding
                    // (§4 no-padding rule).
                    self.cursor += chunk.advance_total();
                    self.next_video_pts += 1;
                }
                ChunkKind::Audio => {
                    // Read only the 8-byte §4b preamble to update the
                    // cumulative sample count; skip the rest of the
                    // body via Seek so we don't pull MB-sized audio
                    // bodies (none exist in AMV, but the principle
                    // holds for the seek hot path).
                    let mut preamble = [0u8; 8];
                    let preamble_len = preamble.len().min(chunk.size as usize);
                    if preamble_len > 0 {
                        self.reader
                            .read_exact(&mut preamble[..preamble_len])
                            .map_err(|e| {
                                Error::other(format!("amv: seek_to audio preamble short read: {e}"))
                            })?;
                    }
                    let block_samples = if preamble_len >= 8 {
                        u32::from_le_bytes([preamble[4], preamble[5], preamble[6], preamble[7]])
                            as i64
                    } else {
                        0
                    };
                    if stream_index == STREAM_INDEX_AUDIO && self.next_audio_pts >= requested_pts {
                        // We've already nibbled at the body; rewind the
                        // reader to the start of this chunk header so
                        // `next_packet` re-reads it cleanly. The cursor
                        // is the source of truth.
                        return Ok(self.next_audio_pts);
                    }
                    self.cursor += chunk.advance_total();
                    self.next_audio_pts += block_samples;
                    self.next_audio_blocks += 1;
                }
                ChunkKind::Other(other) => {
                    return Err(Error::invalid(format!(
                        "amv: unexpected chunk tag {:?} during seek_to at offset {:#x}",
                        std::str::from_utf8(&other).unwrap_or("?"),
                        self.cursor
                    )));
                }
            }
        }
    }
}

impl AmvDemuxer {
    /// Binary-search backed implementation of [`Demuxer::seek_to`] used
    /// once [`build_chunk_index`](Self::build_chunk_index) has populated
    /// the in-memory index. Finds the first index entry on the requested
    /// stream whose pre-emit PTS is `>= requested_pts`, snaps the
    /// walker's cursor + PTS counters to that entry, and returns the
    /// PTS that the next [`Demuxer::next_packet`] call will emit. When
    /// the request is past the last chunk on the stream, the walker
    /// lands at EOF and reports the running PTS.
    fn seek_to_via_index(&mut self, stream_index: u32, requested_pts: i64) -> Result<i64> {
        let index = self
            .chunk_index
            .as_ref()
            .expect("seek_to_via_index requires chunk_index to be populated");
        let target_kind = match stream_index {
            STREAM_INDEX_VIDEO => ChunkKind::Video,
            STREAM_INDEX_AUDIO => ChunkKind::Audio,
            _ => unreachable!(),
        };
        // Linear pass over the index — N is small (a few thousand for
        // the longest-known AMV files) and we want the first entry on
        // the *correct stream* whose pre-emit PTS reaches the target.
        // A pure binary search would need a per-stream subview; the
        // linear pass over the cached vec is O(N) memory-loads with
        // no disk I/O which is dramatically faster than the original
        // disk-walking seek path.
        let mut target_entry: Option<&ChunkIndexEntry> = None;
        for entry in index.iter() {
            if entry.kind != target_kind {
                continue;
            }
            let entry_pts = match target_kind {
                ChunkKind::Video => entry.video_pts_before,
                ChunkKind::Audio => entry.audio_pts_before,
                ChunkKind::Other(_) => unreachable!(),
            };
            if entry_pts >= requested_pts {
                target_entry = Some(entry);
                break;
            }
        }
        match target_entry {
            Some(entry) => {
                self.cursor = entry.file_offset;
                self.next_video_pts = entry.video_pts_before;
                self.next_audio_pts = entry.audio_pts_before;
                self.next_audio_blocks = entry.audio_blocks_before;
                self.eof = false;
                self.reader
                    .seek(SeekFrom::Start(self.cursor))
                    .map_err(|e| Error::other(format!("amv: seek_to_via_index: {e}")))?;
                Ok(match target_kind {
                    ChunkKind::Video => entry.video_pts_before,
                    ChunkKind::Audio => entry.audio_pts_before,
                    ChunkKind::Other(_) => unreachable!(),
                })
            }
            None => {
                // Request is past the last chunk on the stream.
                // Snap to EOF and report the post-walk PTS counters
                // recorded by the index walker (= the last entry's
                // `*_pts_before` plus whatever that chunk would have
                // emitted; we just sum up the *_before of the last
                // entry of either kind and step past it).
                if let Some(last) = index.last() {
                    self.cursor = last.file_offset + 8; // step past the header.
                    self.next_video_pts =
                        last.video_pts_before + if last.kind == ChunkKind::Video { 1 } else { 0 };
                    // For audio, the "after" PTS is the next entry's
                    // before (chunks alternate) or, if last is itself
                    // audio, the running counter from the index walk
                    // post-last. We can't recover the last block's
                    // sample count from the entry alone, so for the
                    // EOF case we land at the running counter recorded
                    // at the entry: audio_pts_before is fine because
                    // any PTS at-or-beyond it satisfies the contract
                    // ("first chunk whose cumulative ... reaches or
                    // exceeds requested_pts" - if none does, EOF is
                    // the answer).
                    self.next_audio_pts = last.audio_pts_before;
                    // The audio *block* count, unlike the sample count,
                    // is recoverable exactly from the entry: step past
                    // the last chunk by adding one block iff it is an
                    // `01wb`. This keeps `audio_blocks_emitted` (and so
                    // the §4 `movi_interleave_balanced` cross-check)
                    // correct at a past-end seek — both counters then
                    // equal the full-drain totals.
                    self.next_audio_blocks = last.audio_blocks_before
                        + if last.kind == ChunkKind::Audio { 1 } else { 0 };
                }
                self.eof = true;
                Ok(match stream_index {
                    STREAM_INDEX_VIDEO => self.next_video_pts,
                    STREAM_INDEX_AUDIO => self.next_audio_pts,
                    _ => unreachable!(),
                })
            }
        }
    }

    fn read_video_packet(&mut self, chunk: ChunkHeader) -> Result<Packet> {
        let mut data = vec![0u8; chunk.size as usize];
        if self.reader.read_exact(&mut data).is_err() {
            // The 8-byte chunk header parsed cleanly but the body
            // didn't fit in the file — the device cut off mid-frame.
            // Drop the partial frame and surface a graceful EOF; the
            // truncation flag lets callers tell apart "drained
            // through `AMV_END_`" from "device died mid-write".
            self.eof = true;
            self.truncated = true;
            return Err(Error::Eof);
        }
        // Advance cursor by exactly 8 + size — NO padding (§4
        // "no-padding rule").
        self.cursor += chunk.advance_total();
        let pts = self.next_video_pts;
        self.next_video_pts += 1;
        let mut pkt = Packet::new(STREAM_INDEX_VIDEO, self.streams[0].time_base, data);
        pkt.pts = Some(pts);
        pkt.dts = Some(pts);
        pkt.duration = Some(1);
        // Every AMV video frame is intra-only (§4a): no inter
        // prediction across frames, every frame is a keyframe.
        pkt.flags.keyframe = true;
        Ok(pkt)
    }

    fn read_audio_packet(&mut self, chunk: ChunkHeader) -> Result<Packet> {
        let mut data = vec![0u8; chunk.size as usize];
        if self.reader.read_exact(&mut data).is_err() {
            // Mirror the video-side body-truncation recovery: an
            // 8-byte header parsed cleanly but the body fell off the
            // end. Drop the partial block and surface a graceful EOF
            // with the truncation flag set.
            self.eof = true;
            self.truncated = true;
            return Err(Error::Eof);
        }
        self.cursor += chunk.advance_total();
        // A complete `01wb` chunk landed — count it as one audio block
        // for the §4 1:1-pairing cross-check (see
        // `movi_interleave_balanced`). The block count steps by exactly
        // one regardless of the preamble's decoded-sample contribution
        // below, mirroring the video side's per-chunk `+1`.
        self.next_audio_blocks += 1;
        // §4b: the first 8 bytes of every `01wb` payload are an
        // 8-byte preamble — `u32` per-block state + `u32` decoded
        // sample count. Use the decoded-sample-count to drive PTS
        // and packet duration.
        let pts = self.next_audio_pts;
        let mut duration = 0i64;
        if chunk.size >= 8 {
            let decoded_samples = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
            duration = decoded_samples as i64;
            self.next_audio_pts += duration;
        }
        let mut pkt = Packet::new(STREAM_INDEX_AUDIO, self.streams[1].time_base, data);
        pkt.pts = Some(pts);
        pkt.dts = Some(pts);
        if duration > 0 {
            pkt.duration = Some(duration);
        }
        // ADPCM-style audio is also keyframe-equivalent (no inter
        // dependency across blocks beyond the per-block preamble's
        // initial state).
        pkt.flags.keyframe = true;
        Ok(pkt)
    }
}

/// Reduce a `TimeBase = 1/N` into "microseconds per tick" so the
/// stream-level `duration_micros()` can be converted into a count of
/// ticks for [`StreamInfo::duration`]. Stays in integer math.
fn micros_per_tick(tb: TimeBase) -> i64 {
    let r = tb.as_rational();
    if r.num == 0 || r.den == 0 {
        1
    } else {
        // 1 tick = (num / den) seconds = (num * 1_000_000 / den) µs.
        (1_000_000 * r.num) / r.den
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::tests::build_synthetic_prelude;
    use crate::parse::{AUDIO_CHUNK_TAG, VIDEO_CHUNK_TAG};
    use std::io::Cursor;

    /// Build a complete, synthetic AMV file: prelude + N pairs of
    /// `00dc` / `01wb` chunks + `AMV_END_` trailer. Used to exercise
    /// the demuxer end-to-end without depending on the staged
    /// fixture's docs/.
    fn build_synthetic_file(n_pairs: usize) -> Vec<u8> {
        build_synthetic_file_with_duration(n_pairs, 0x0000_0121)
    }

    /// Same as [`build_synthetic_file`] but with a caller-chosen
    /// `amvh +0x34` packed-byte duration so the §2-header↔§4-walk
    /// cross-check ([`AmvDemuxer::duration_consistent_with_drained_frames`])
    /// can be exercised for both a matching and a deliberately
    /// mismatched header.
    fn build_synthetic_file_with_duration(n_pairs: usize, duration_packed: u32) -> Vec<u8> {
        let mut buf = build_synthetic_prelude(128, 96, 12, duration_packed, 22_050);
        for i in 0..n_pairs {
            // Video: 4-byte payload with the iteration index so we
            // can tell frames apart.
            let video_body = (i as u32).to_le_bytes().to_vec();
            buf.extend_from_slice(&VIDEO_CHUNK_TAG);
            buf.extend_from_slice(&(video_body.len() as u32).to_le_bytes());
            buf.extend_from_slice(&video_body);
            // Audio: 12-byte payload — 8-byte preamble (state=0,
            // decoded_samples=1837) + 4 nibble-coded bytes.
            let mut audio_body = Vec::with_capacity(12);
            audio_body.extend_from_slice(&0u32.to_le_bytes()); // per-block state
            audio_body.extend_from_slice(&1837u32.to_le_bytes()); // decoded samples
            audio_body.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
            buf.extend_from_slice(&AUDIO_CHUNK_TAG);
            buf.extend_from_slice(&(audio_body.len() as u32).to_le_bytes());
            buf.extend_from_slice(&audio_body);
        }
        buf.extend_from_slice(&AMV_END_TRAILER);
        buf
    }

    #[test]
    fn open_parses_prelude_and_initialises_cursor() {
        let buf = build_synthetic_file(3);
        let d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        assert_eq!(d.streams().len(), 2);
        assert_eq!(d.streams()[0].params.width, Some(128));
        assert_eq!(d.streams()[0].params.height, Some(96));
        assert_eq!(d.streams()[1].params.sample_rate, Some(22_050));
        assert_eq!(d.streams()[1].params.channels, Some(1));
        // The decoded §4b PCM is 16-bit signed mono — the audio stream
        // params describe the decode output, not the on-disk nibbles.
        assert_eq!(
            d.streams()[1].params.sample_format,
            Some(oxideav_core::SampleFormat::S16)
        );
        assert_eq!(
            d.streams()[1].params.channel_layout,
            Some(oxideav_core::ChannelLayout::from_count(1))
        );
        assert_eq!(d.cursor(), 0x13C);
        assert_eq!(d.duration_micros(), Some(93_000_000));
    }

    #[test]
    fn demuxer_declares_direct_codec_ids_resolvable_through_registry() {
        // The video stream must declare `amv_video` (not generic `mjpeg`)
        // and the audio stream `adpcm_amv`, so a RuntimeContext that has
        // `register`-ed this crate resolves a working decoder for each
        // stream straight from the demuxer's StreamInfo.
        let buf = build_synthetic_file(3);
        let d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        assert_eq!(
            d.streams()[0].params.codec_id.as_str(),
            crate::VIDEO_DIRECT_CODEC_ID
        );
        assert_eq!(d.streams()[1].params.codec_id.as_str(), AUDIO_CODEC_ID);

        let mut ctx = oxideav_core::RuntimeContext::new();
        crate::register(&mut ctx);
        assert!(
            ctx.codecs.first_decoder(&d.streams()[0].params).is_ok(),
            "video stream params resolve a registered decoder"
        );
        assert!(
            ctx.codecs.first_decoder(&d.streams()[1].params).is_ok(),
            "audio stream params resolve a registered decoder"
        );
    }

    #[test]
    fn video_frames_emitted_tracks_drained_count() {
        // 12 pairs: 12 video chunks, 12 audio chunks, then trailer.
        let buf = build_synthetic_file(12);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        assert_eq!(d.video_frames_emitted(), 0, "no frames before any drain");
        // Emit one V/A pair → one video frame counted.
        let _v0 = d.next_packet().expect("v0");
        assert_eq!(d.video_frames_emitted(), 1, "one video chunk emitted");
        let _a0 = d.next_packet().expect("a0");
        assert_eq!(
            d.video_frames_emitted(),
            1,
            "audio chunk does not bump the video count"
        );
        // Drain the rest to EOF.
        loop {
            match d.next_packet() {
                Ok(_) => {}
                Err(Error::Eof) => break,
                Err(e) => panic!("walk error: {e:?}"),
            }
        }
        assert_eq!(
            d.video_frames_emitted(),
            12,
            "all 12 video chunks counted after a full drain"
        );
    }

    #[test]
    fn duration_cross_check_matches_when_header_agrees_with_walk() {
        // §2 worked example: at 12 fps, 1116 video frames pack to
        // 1116 / 12 = 93 s = 1:33 = packed 0x0000_0121. Build a file
        // whose header carries exactly that duration and whose walk
        // drains exactly 1116 video frames, and confirm the demuxer's
        // header↔walk cross-check agrees after a full drain.
        let buf = build_synthetic_file_with_duration(1116, 0x0000_0121);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        loop {
            match d.next_packet() {
                Ok(_) => {}
                Err(Error::Eof) => break,
                Err(e) => panic!("walk error: {e:?}"),
            }
        }
        assert_eq!(d.video_frames_emitted(), 1116);
        assert!(
            d.duration_consistent_with_drained_frames(),
            "§2 packed 1:33 must agree with 1116 frames at 12 fps"
        );
    }

    #[test]
    fn duration_cross_check_rejects_when_header_disagrees_with_walk() {
        // Header still declares the full 1:33 duration (0x0000_0121),
        // but the file only carries 3 video frames — the shape of a
        // truncated / re-stamped file whose header was not corrected.
        // 3 / 12 = 0 s, which is NOT 1:33, so the cross-check fails.
        let buf = build_synthetic_file_with_duration(3, 0x0000_0121);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        loop {
            match d.next_packet() {
                Ok(_) => {}
                Err(Error::Eof) => break,
                Err(e) => panic!("walk error: {e:?}"),
            }
        }
        assert_eq!(d.video_frames_emitted(), 3);
        assert!(
            !d.duration_consistent_with_drained_frames(),
            "header's 1:33 must not agree with only 3 drained frames"
        );
    }

    #[test]
    fn audio_blocks_emitted_tracks_drained_count() {
        // §4: every `01wb` chunk bumps the audio-block counter by one;
        // a `00dc` chunk leaves it untouched (it bumps the video count).
        let buf = build_synthetic_file(12);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        assert_eq!(d.audio_blocks_emitted(), 0, "no blocks before any drain");
        // First `00dc`: video count goes to 1, audio blocks stay 0 —
        // §4 video-first cadence means the video count leads by one.
        let _v0 = d.next_packet().expect("v0");
        assert_eq!(d.video_frames_emitted(), 1);
        assert_eq!(
            d.audio_blocks_emitted(),
            0,
            "video chunk does not bump the audio-block count"
        );
        // First `01wb`: audio block counted, counts now equal.
        let _a0 = d.next_packet().expect("a0");
        assert_eq!(d.audio_blocks_emitted(), 1, "one audio block emitted");
        // Drain to EOF.
        loop {
            match d.next_packet() {
                Ok(_) => {}
                Err(Error::Eof) => break,
                Err(e) => panic!("walk error: {e:?}"),
            }
        }
        assert_eq!(
            d.audio_blocks_emitted(),
            12,
            "all 12 audio blocks counted after a full drain"
        );
        assert_eq!(
            d.video_frames_emitted(),
            12,
            "all 12 video frames counted after a full drain"
        );
    }

    #[test]
    fn interleave_balanced_is_true_only_at_pair_boundaries_then_holds_at_eof() {
        // §4 strict 1:1 video-first: during the walk the video count
        // leads the audio count by at most one, so `balanced` is true
        // exactly at pair boundaries (start, and after each `01wb`).
        let buf = build_synthetic_file(3);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        // At the start (0 == 0) — balanced.
        assert!(d.movi_interleave_balanced(), "balanced before any drain");
        // After the first `00dc` (1 vs 0) — NOT balanced (video leads).
        let _v0 = d.next_packet().expect("v0");
        assert!(
            !d.movi_interleave_balanced(),
            "video leads by one after a `00dc`"
        );
        // After its paired `01wb` (1 vs 1) — balanced again.
        let _a0 = d.next_packet().expect("a0");
        assert!(
            d.movi_interleave_balanced(),
            "balanced at the pair boundary after `01wb`"
        );
        // Drain the rest to a clean trailer EOF.
        loop {
            match d.next_packet() {
                Ok(_) => {}
                Err(Error::Eof) => break,
                Err(e) => panic!("walk error: {e:?}"),
            }
        }
        assert!(
            d.trailer_offset().is_some(),
            "clean trailer-bounded drain expected"
        );
        assert!(
            d.movi_interleave_balanced(),
            "§4 1:1 pairing holds over the whole payload (3 == 3)"
        );
        assert_eq!(d.video_frames_emitted(), 3);
        assert_eq!(d.audio_blocks_emitted(), 3);
    }

    #[test]
    fn interleave_unbalanced_when_truncated_after_trailing_video() {
        // §4 reports a missing trailing audio block as the unpaired
        // -video signal. Build a complete 3-pair file, then drop the
        // trailer + the final `01wb` + the final `00dc`'s body to leave
        // a power-cut shape ending after a `00dc` header with no body.
        // The demuxer's body-truncation recovery drops the partial
        // video frame and surfaces a graceful EOF, leaving 2 complete
        // video frames and 2 complete audio blocks — still balanced.
        // To force an imbalance we truncate so the LAST complete chunk
        // is a `00dc` with a fully-read body but no following `01wb`.
        let full = build_synthetic_file(3);
        // Layout per pair: `00dc`(8+4) + `01wb`(8+12) = 32 bytes; the
        // prelude is 0x13C. Keep the prelude + 2 full pairs + the third
        // pair's complete `00dc` (12 bytes), dropping its `01wb` and the
        // trailer. That leaves 3 video frames but only 2 audio blocks.
        let cut = 0x13C + 2 * 32 + 12;
        let truncated = full[..cut].to_vec();
        let mut d = AmvDemuxer::open(Cursor::new(truncated)).expect("open");
        loop {
            match d.next_packet() {
                Ok(_) => {}
                Err(Error::Eof) => break,
                Err(e) => panic!("walk error: {e:?}"),
            }
        }
        assert!(d.is_truncated(), "no trailer landed → truncated drain");
        assert!(d.trailer_offset().is_none());
        assert_eq!(d.video_frames_emitted(), 3, "3 complete video frames");
        assert_eq!(d.audio_blocks_emitted(), 2, "only 2 complete audio blocks");
        assert!(
            !d.movi_interleave_balanced(),
            "trailing unpaired video frame leaves the interleave unbalanced"
        );
        assert_eq!(
            d.video_frames_emitted() - d.audio_blocks_emitted(),
            1,
            "exactly one video frame left unpaired by the cut"
        );
    }

    #[test]
    fn audio_blocks_rewind_with_cursor_on_backward_seek() {
        // The audio-block counter must rewind / fast-forward with the
        // cursor exactly like the PTS counters so `balanced` stays
        // meaningful across seeks (linear path — no index built).
        let buf = build_synthetic_file(5);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        // Forward seek to video frame 4 → 4 video + 4 audio walked past.
        d.seek_to(0, 4).expect("seek fwd");
        assert_eq!(d.video_frames_emitted(), 4);
        assert_eq!(d.audio_blocks_emitted(), 4);
        assert!(d.movi_interleave_balanced(), "pair boundary at frame 4");
        // Backward seek to frame 1 → counters rewind together.
        d.seek_to(0, 1).expect("seek back");
        assert_eq!(d.video_frames_emitted(), 1);
        assert_eq!(d.audio_blocks_emitted(), 1);
        assert!(d.movi_interleave_balanced(), "pair boundary at frame 1");
    }

    #[test]
    fn audio_blocks_track_indexed_seek_and_past_end() {
        // The indexed seek path snaps the audio-block counter to the
        // target entry's `audio_blocks_before`, and a past-end indexed
        // seek lands both counters on the full-drain totals.
        let buf = build_synthetic_file(5);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        d.build_chunk_index().expect("index");
        // Indexed seek to video frame 3 — 3 complete pairs precede it.
        d.seek_to(0, 3).expect("indexed seek");
        assert_eq!(d.video_frames_emitted(), 3);
        assert_eq!(d.audio_blocks_emitted(), 3);
        assert!(d.movi_interleave_balanced());
        // Past-end indexed seek — both counters reach the 5/5 totals.
        d.seek_to(0, 1_000_000).expect("indexed past-end");
        assert_eq!(d.video_frames_emitted(), 5);
        assert_eq!(d.audio_blocks_emitted(), 5);
        assert!(
            d.movi_interleave_balanced(),
            "past-end indexed seek lands on the balanced full-drain totals"
        );
    }

    #[test]
    fn chunk_index_records_audio_blocks_before_monotonically() {
        // Each indexed `01wb` entry's `audio_blocks_before` equals the
        // number of audio blocks ahead of it; video entries share the
        // count of the block before them (the §4 video-first cadence).
        let buf = build_synthetic_file(4);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        d.build_chunk_index().expect("index");
        let idx = d.chunk_index().expect("built");
        // 4 pairs → 8 entries, V A V A V A V A.
        let mut seen_audio = 0u64;
        for entry in idx {
            assert_eq!(
                entry.audio_blocks_before, seen_audio,
                "audio_blocks_before must equal blocks emitted ahead of {entry:?}"
            );
            if entry.kind == ChunkKind::Audio {
                seen_audio += 1;
            }
        }
        assert_eq!(seen_audio, 4, "four audio blocks indexed");
    }

    #[test]
    fn next_packet_walks_video_audio_pairs_until_trailer() {
        let buf = build_synthetic_file(2);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        // Expect: V(0), A(0), V(1), A(1), then EOF.
        let p0 = d.next_packet().expect("v0");
        assert_eq!(p0.stream_index, 0);
        assert_eq!(p0.pts, Some(0));
        assert_eq!(p0.data, 0u32.to_le_bytes().to_vec());
        assert!(p0.flags.keyframe);
        let a0 = d.next_packet().expect("a0");
        assert_eq!(a0.stream_index, 1);
        assert_eq!(a0.pts, Some(0));
        assert_eq!(a0.duration, Some(1837));
        let p1 = d.next_packet().expect("v1");
        assert_eq!(p1.stream_index, 0);
        assert_eq!(p1.pts, Some(1));
        assert_eq!(p1.data, 1u32.to_le_bytes().to_vec());
        let a1 = d.next_packet().expect("a1");
        assert_eq!(a1.stream_index, 1);
        assert_eq!(a1.pts, Some(1837));
        // Next call should hit the AMV_END_ trailer and return EOF.
        let err = d.next_packet().unwrap_err();
        match err {
            Error::Eof => {}
            other => panic!("expected Eof, got {other:?}"),
        }
        // Subsequent calls keep returning EOF.
        assert!(matches!(d.next_packet().unwrap_err(), Error::Eof));
    }

    #[test]
    fn next_packet_advances_with_no_byte_padding() {
        // Build a synthetic AMV with intentionally **odd-sized**
        // payloads (§4 "no-padding rule"). If the demuxer pads to
        // an even boundary the cursor desyncs and the second chunk
        // header reads garbage.
        let mut buf = build_synthetic_prelude(128, 96, 12, 0x0000_0121, 22_050);
        let video_body = vec![0xDE, 0xAD, 0xBE]; // 3 bytes (odd)
        buf.extend_from_slice(&VIDEO_CHUNK_TAG);
        buf.extend_from_slice(&(video_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(&video_body);
        let mut audio_body = Vec::with_capacity(15); // 8 (hdr) + 7 = odd
        audio_body.extend_from_slice(&0u32.to_le_bytes());
        audio_body.extend_from_slice(&100u32.to_le_bytes());
        audio_body.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x99]);
        buf.extend_from_slice(&AUDIO_CHUNK_TAG);
        buf.extend_from_slice(&(audio_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(&audio_body);
        buf.extend_from_slice(&AMV_END_TRAILER);

        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        let v = d.next_packet().expect("video");
        assert_eq!(v.data, vec![0xDE, 0xAD, 0xBE]);
        let a = d.next_packet().expect("audio");
        assert_eq!(a.data.len(), 15);
        assert!(matches!(d.next_packet().unwrap_err(), Error::Eof));
    }

    #[test]
    fn seek_to_video_rewinds_to_start_then_walks_forward() {
        // Build 5 (video, audio) pairs, then seek_to the video frame
        // at PTS = 3. Expect: cursor lands on the chunk header for
        // frame 3, and the next packet emitted on stream 0 carries
        // pts == 3 and the recorded payload bytes.
        let buf = build_synthetic_file(5);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        // First, advance past all packets to land at EOF (so
        // current_pts > requested_pts and seek_to must rewind).
        loop {
            match d.next_packet() {
                Ok(_) => {}
                Err(Error::Eof) => break,
                Err(e) => panic!("walk error: {e:?}"),
            }
        }
        let landed = d.seek_to(0, 3).expect("seek_to(0, 3)");
        assert_eq!(landed, 3);
        let p = d.next_packet().expect("packet after seek");
        assert_eq!(p.stream_index, 0);
        assert_eq!(p.pts, Some(3));
        assert_eq!(p.data, 3u32.to_le_bytes().to_vec());
    }

    #[test]
    fn seek_to_video_forward_does_not_rewind() {
        // Build 5 pairs, advance through the first video frame, then
        // seek forward to video PTS = 4. Verify the next stream-0
        // packet carries pts == 4 — the seek walks forward from the
        // current cursor without rewinding.
        let buf = build_synthetic_file(5);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        let _v0 = d.next_packet().expect("v0");
        let _a0 = d.next_packet().expect("a0");
        let landed = d.seek_to(0, 4).expect("seek forward");
        assert_eq!(landed, 4);
        let p = d.next_packet().expect("v4");
        assert_eq!(p.stream_index, 0);
        assert_eq!(p.pts, Some(4));
        assert_eq!(p.data, 4u32.to_le_bytes().to_vec());
    }

    #[test]
    fn seek_to_audio_uses_cumulative_sample_count() {
        // §4b: audio PTS = running decoded-sample count from each
        // chunk's 8-byte preamble. Synthetic file writes 1837 samples
        // per chunk, so after 2 audio chunks the running PTS is 3674.
        // A seek to PTS = 3674 must land on chunk index 2 (0-based).
        let buf = build_synthetic_file(5);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        let landed = d.seek_to(1, 3674).expect("seek_to audio 3674");
        assert_eq!(landed, 3674);
        // The next packet on the audio stream should carry pts == 3674
        // and the audio body. (We may emit video chunks in between
        // because they share the same `movi` walk.)
        loop {
            let p = d.next_packet().expect("packet");
            if p.stream_index == 1 {
                assert_eq!(p.pts, Some(3674));
                break;
            }
        }
    }

    #[test]
    fn seek_to_past_end_lands_at_eof() {
        // A seek beyond the last frame returns the running PTS and
        // sets the demuxer to EOF.
        let buf = build_synthetic_file(2);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        let landed = d.seek_to(0, 1_000_000).expect("seek past end");
        // 2 video frames walked → PTS counter is 2.
        assert_eq!(landed, 2);
        assert!(matches!(d.next_packet().unwrap_err(), Error::Eof));
    }

    #[test]
    fn seek_to_zero_resets_to_start_of_movi() {
        // After walking some packets, seek_to(stream, 0) must rewind to
        // movi_start and the next emitted packet's PTS must be 0.
        let buf = build_synthetic_file(3);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        let _v0 = d.next_packet().expect("v0");
        let _a0 = d.next_packet().expect("a0");
        let _v1 = d.next_packet().expect("v1");
        let landed = d.seek_to(0, 0).expect("seek to 0");
        assert_eq!(landed, 0);
        let v = d.next_packet().expect("v0 after seek");
        assert_eq!(v.stream_index, 0);
        assert_eq!(v.pts, Some(0));
        assert_eq!(v.data, 0u32.to_le_bytes().to_vec());
    }

    #[test]
    fn seek_to_rejects_invalid_stream_or_negative_pts() {
        let buf = build_synthetic_file(2);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        // Negative PTS.
        assert!(matches!(
            d.seek_to(0, -1).unwrap_err(),
            Error::InvalidData(_)
        ));
        // Out-of-range stream index.
        assert!(matches!(
            d.seek_to(99, 0).unwrap_err(),
            Error::InvalidData(_)
        ));
    }

    #[test]
    fn open_rejects_truncated_prelude() {
        let buf = vec![0u8; 16];
        let err = AmvDemuxer::open(Cursor::new(buf)).unwrap_err();
        // Short read → Io; corrupt prelude → InvalidData. Either is
        // acceptable for this case.
        assert!(matches!(
            err,
            AmvDemuxerError::Io(_) | AmvDemuxerError::InvalidData(_)
        ));
    }

    #[test]
    fn open_rejects_invalid_form_type() {
        let mut buf = build_synthetic_prelude(128, 96, 12, 0x0000_0121, 22_050);
        buf[8..12].copy_from_slice(b"AVI ");
        let err = AmvDemuxer::open(Cursor::new(buf)).unwrap_err();
        assert!(matches!(err, AmvDemuxerError::InvalidData(_)));
    }

    /// Real fixture test: open the staged `comedian.amv` and walk
    /// the whole file, validating the chunk counts + headline
    /// `amvh` fields recorded in `docs/container/amv/amv-container-trace.md`.
    ///
    /// The fixture is searched in two locations so the test runs
    /// both when the crate is built standalone (with the fixture
    /// staged under `tests/fixtures/`) and from inside the
    /// workspace tree (which keeps the canonical copy under
    /// `docs/container/amv/fixtures/`).
    #[test]
    fn comedian_fixture_walks_to_eof_with_expected_counts() {
        let crate_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/comedian.amv");
        let workspace_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/container/amv/fixtures/comedian.amv");
        let path = if crate_path.exists() {
            crate_path
        } else if workspace_path.exists() {
            workspace_path
        } else {
            // Fixture is provided by a workspace-level setup step;
            // skip cleanly when not staged so this test stays
            // green in environments without the binary blob.
            eprintln!(
                "skipping comedian fixture test: not found at {} or {}",
                crate_path.display(),
                workspace_path.display()
            );
            return;
        };
        let f = std::fs::File::open(&path).expect("open fixture");
        let mut d = AmvDemuxer::open(std::io::BufReader::new(f)).expect("open comedian.amv");
        assert_eq!(d.header().width, 128);
        assert_eq!(d.header().height, 96);
        assert_eq!(d.header().fps, 12);
        assert_eq!(d.header().micros_per_frame, 83_333);
        assert_eq!(d.header().duration().seconds, 0x21);
        assert_eq!(d.header().duration().minutes, 0x01);
        assert_eq!(d.header().duration().hours, 0);
        assert_eq!(d.audio_format().samples_per_sec, 22_050);
        assert_eq!(d.audio_format().channels, 1);

        // Walk to EOF.
        let mut n_video = 0u32;
        let mut n_audio = 0u32;
        let mut first_video_size = None;
        loop {
            match d.next_packet() {
                Ok(p) if p.stream_index == 0 => {
                    if first_video_size.is_none() {
                        first_video_size = Some(p.data.len());
                    }
                    n_video += 1;
                }
                Ok(p) if p.stream_index == 1 => {
                    n_audio += 1;
                }
                Ok(_) => panic!("unexpected stream index"),
                Err(Error::Eof) => break,
                Err(other) => panic!("walk error: {other:?}"),
            }
        }
        assert_eq!(n_video, 1116, "expected 1116 video chunks");
        assert_eq!(n_audio, 1116, "expected 1116 audio chunks");
        // §4 worked example: first video chunk is 1633 bytes (size
        // 0x661).
        assert_eq!(first_video_size, Some(1633));

        // §4 no-padding proof: the `8 + size` walk lands the `AMV_END_`
        // trailer at file offset 0x348E31 — exactly 8 bytes before the
        // 0x348E39 EOF.
        let stream_len = std::fs::metadata(&path).expect("metadata").len();
        assert_eq!(stream_len, 0x348E39, "comedian.amv is 0x348E39 bytes");
        assert_eq!(
            d.trailer_offset(),
            Some(0x348E31),
            "§4 trailer lands at 0x348E31 after the no-padding walk"
        );
        assert_eq!(
            d.trailer_matches_eof(stream_len),
            Some(true),
            "§4c trailer is the final 8 bytes (stream_len - 8)"
        );
        assert!(!d.is_truncated(), "complete fixture is not truncated");
    }

    /// Real-fixture cross-check: walk `comedian.amv` to EOF and confirm
    /// the §2 `amvh` packed duration agrees with the §4 `movi`-walk
    /// video-frame count, the trace's central header↔payload invariant
    /// ("1116 frames ÷ 12 fps = 93 s = 1:33", §2).
    #[test]
    fn comedian_fixture_duration_agrees_with_drained_frame_count() {
        let crate_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/comedian.amv");
        let workspace_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/container/amv/fixtures/comedian.amv");
        let path = if crate_path.exists() {
            crate_path
        } else if workspace_path.exists() {
            workspace_path
        } else {
            eprintln!("skipping comedian duration cross-check: fixture not found");
            return;
        };
        let f = std::fs::File::open(&path).expect("open comedian.amv");
        let mut d =
            AmvDemuxer::open(std::io::BufReader::new(f)).expect("open comedian.amv demuxer");
        // Drain to a clean trailer-bounded EOF.
        loop {
            match d.next_packet() {
                Ok(_) => {}
                Err(Error::Eof) => break,
                Err(e) => panic!("walk error: {e:?}"),
            }
        }
        assert!(!d.is_truncated(), "complete fixture drains via the trailer");
        assert_eq!(
            d.video_frames_emitted(),
            1116,
            "§4 walk drains 1116 video frames"
        );
        assert!(
            d.duration_consistent_with_drained_frames(),
            "§2 packed 1:33 must agree with 1116 frames at 12 fps"
        );
    }

    /// Real-fixture seek_to test: walk to EOF, then seek back to video
    /// frame 500 on `comedian.amv`. Verify the next video packet has
    /// `pts == 500` and the payload still starts with the JPEG SOI
    /// marker (`FF D8`) — the trace doc §4a records every video chunk
    /// as a self-contained `FF D8 … FF D9` JPEG.
    #[test]
    fn comedian_fixture_seek_to_video_frame_500() {
        let crate_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/comedian.amv");
        let workspace_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/container/amv/fixtures/comedian.amv");
        let path = if crate_path.exists() {
            crate_path
        } else if workspace_path.exists() {
            workspace_path
        } else {
            eprintln!(
                "skipping comedian seek_to test: fixture not found at {} or {}",
                crate_path.display(),
                workspace_path.display()
            );
            return;
        };
        let f = std::fs::File::open(&path).expect("open fixture");
        let mut d = AmvDemuxer::open(std::io::BufReader::new(f)).expect("open comedian.amv");
        // Drain forward first so seek_to has to rewind.
        loop {
            match d.next_packet() {
                Ok(_) => {}
                Err(Error::Eof) => break,
                Err(e) => panic!("walk error: {e:?}"),
            }
        }
        // Seek back to video frame 500.
        let landed = d.seek_to(0, 500).expect("seek_to comedian frame 500");
        assert_eq!(landed, 500);
        // Next video packet must be frame 500 and start with JPEG SOI.
        loop {
            let p = d.next_packet().expect("packet after seek");
            if p.stream_index == 0 {
                assert_eq!(p.pts, Some(500));
                assert!(
                    p.data.len() >= 2 && p.data[0] == 0xFF && p.data[1] == 0xD8,
                    "comedian frame 500 must start with JPEG SOI"
                );
                break;
            }
        }
    }

    // ─────────────── chunk-index cache tests ───────────────

    #[test]
    fn build_chunk_index_records_every_chunk_in_order() {
        // 4 (video, audio) pairs → 8 chunks in alternating order.
        // Build the index and verify each entry's offset / kind /
        // pre-emit PTS counters are correct.
        let buf = build_synthetic_file(4);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        d.build_chunk_index().expect("build_chunk_index");
        let idx = d.chunk_index().expect("index populated");
        assert_eq!(idx.len(), 8, "4 video + 4 audio chunks = 8");
        // Strict 1:1 alternation, video-first.
        let kinds: Vec<_> = idx.iter().map(|e| e.kind).collect();
        for (i, k) in kinds.iter().enumerate() {
            let expected = if i % 2 == 0 {
                ChunkKind::Video
            } else {
                ChunkKind::Audio
            };
            assert_eq!(*k, expected, "entry {i} kind");
        }
        // Per-stream pre-emit PTS counters: video chunk i has
        // video_pts_before == i; audio chunk i has
        // audio_pts_before == i * 1837 (the synthetic per-block sample
        // count).
        for i in 0..4 {
            let v = &idx[2 * i];
            assert_eq!(v.video_pts_before, i as i64);
            assert_eq!(v.audio_pts_before, (i as i64) * 1837);
            let a = &idx[2 * i + 1];
            assert_eq!(a.video_pts_before, (i + 1) as i64);
            assert_eq!(a.audio_pts_before, (i as i64) * 1837);
        }
        // First entry's file offset is the movi_start.
        assert_eq!(idx[0].file_offset, 0x13C);
    }

    #[test]
    fn build_chunk_index_preserves_walker_state() {
        // build_chunk_index is documented as idempotent + walker-state
        // preserving. Walk past the first packet, build the index, then
        // confirm the next packet still arrives where the caller left
        // off (rather than from movi_start).
        let buf = build_synthetic_file(3);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        let v0 = d.next_packet().expect("v0");
        assert_eq!(v0.pts, Some(0));
        let cursor_before = d.cursor();
        let video_pts_before = d.next_video_pts;
        let audio_pts_before = d.next_audio_pts;
        d.build_chunk_index().expect("build_chunk_index");
        // Walker state should be unchanged.
        assert_eq!(d.cursor(), cursor_before);
        assert_eq!(d.next_video_pts, video_pts_before);
        assert_eq!(d.next_audio_pts, audio_pts_before);
        // And the next packet is the audio chunk that would have
        // followed v0 without the build call.
        let a0 = d.next_packet().expect("a0");
        assert_eq!(a0.stream_index, 1);
        assert_eq!(a0.pts, Some(0));
    }

    #[test]
    fn seek_to_via_index_lands_on_video_frame_3() {
        // Once the index is built, seek_to should take the binary-
        // search path and land exactly on the chunk for video frame 3.
        let buf = build_synthetic_file(5);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        d.build_chunk_index().expect("build_chunk_index");
        // Drain so seek_to has to rewind.
        loop {
            match d.next_packet() {
                Ok(_) => {}
                Err(Error::Eof) => break,
                Err(e) => panic!("walk error: {e:?}"),
            }
        }
        let landed = d.seek_to(0, 3).expect("seek");
        assert_eq!(landed, 3);
        let p = d.next_packet().expect("packet after seek");
        assert_eq!(p.stream_index, 0);
        assert_eq!(p.pts, Some(3));
        assert_eq!(p.data, 3u32.to_le_bytes().to_vec());
    }

    #[test]
    fn seek_to_via_index_audio_uses_cumulative_pts() {
        // Indexed audio seek uses cumulative decoded-sample count
        // exactly like the linear path. After 2 audio chunks the
        // cumulative count is 3674; seek_to(1, 3674) lands there.
        let buf = build_synthetic_file(5);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        d.build_chunk_index().expect("build_chunk_index");
        let landed = d.seek_to(1, 3674).expect("seek");
        assert_eq!(landed, 3674);
        loop {
            let p = d.next_packet().expect("packet");
            if p.stream_index == 1 {
                assert_eq!(p.pts, Some(3674));
                break;
            }
        }
    }

    #[test]
    fn seek_to_via_index_matches_linear_seek_results() {
        // For every PTS the linear seek can land on, the indexed seek
        // must land on the same one. Drives parity between the two
        // code paths so the cache stays trustworthy.
        let buf = build_synthetic_file(5);
        let mut d_linear = AmvDemuxer::open(Cursor::new(buf.clone())).expect("open");
        let mut d_indexed = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        d_indexed.build_chunk_index().expect("build_chunk_index");
        for target_pts in [0i64, 1, 2, 3, 4] {
            let l = d_linear.seek_to(0, target_pts).expect("linear seek");
            let i = d_indexed.seek_to(0, target_pts).expect("indexed seek");
            assert_eq!(l, i, "video pts {target_pts}: linear={l}, indexed={i}");
        }
        for target_pts in [0i64, 1837, 3674, 5511] {
            let l = d_linear.seek_to(1, target_pts).expect("linear seek");
            let i = d_indexed.seek_to(1, target_pts).expect("indexed seek");
            assert_eq!(l, i, "audio pts {target_pts}: linear={l}, indexed={i}");
        }
    }

    #[test]
    fn seek_to_via_index_past_end_lands_at_eof() {
        let buf = build_synthetic_file(2);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        d.build_chunk_index().expect("build_chunk_index");
        let landed = d.seek_to(0, 1_000_000).expect("seek past end");
        assert_eq!(landed, 2);
        assert!(matches!(d.next_packet().unwrap_err(), Error::Eof));
    }

    #[test]
    fn chunk_index_is_none_before_build() {
        let buf = build_synthetic_file(2);
        let d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        assert!(d.chunk_index().is_none());
    }

    #[test]
    fn build_chunk_index_is_idempotent() {
        // Calling twice rebuilds from scratch and produces the same
        // result.
        let buf = build_synthetic_file(3);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        d.build_chunk_index().expect("first build");
        let first: Vec<ChunkIndexEntry> = d.chunk_index().unwrap().to_vec();
        d.build_chunk_index().expect("second build");
        let second: Vec<ChunkIndexEntry> = d.chunk_index().unwrap().to_vec();
        assert_eq!(first, second);
    }

    /// Real-fixture indexed seek: build the index from comedian.amv
    /// and confirm seek_to(0, 500) lands on the same JPEG-SOI-starting
    /// payload the linear test confirms.
    #[test]
    fn comedian_fixture_indexed_seek_matches_jpeg_soi() {
        let crate_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/comedian.amv");
        let workspace_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/container/amv/fixtures/comedian.amv");
        let path = if crate_path.exists() {
            crate_path
        } else if workspace_path.exists() {
            workspace_path
        } else {
            eprintln!(
                "skipping comedian indexed-seek test: fixture not found at {} or {}",
                crate_path.display(),
                workspace_path.display()
            );
            return;
        };
        let f = std::fs::File::open(&path).expect("open fixture");
        let mut d = AmvDemuxer::open(std::io::BufReader::new(f)).expect("open comedian.amv");
        d.build_chunk_index().expect("build index");
        let idx = d.chunk_index().expect("index populated");
        // 1116 video + 1116 audio.
        assert_eq!(idx.len(), 2232);
        // Seek via the index.
        let landed = d.seek_to(0, 500).expect("indexed seek");
        assert_eq!(landed, 500);
        loop {
            let p = d.next_packet().expect("packet after seek");
            if p.stream_index == 0 {
                assert_eq!(p.pts, Some(500));
                assert!(
                    p.data.len() >= 2 && p.data[0] == 0xFF && p.data[1] == 0xD8,
                    "comedian frame 500 must start with JPEG SOI"
                );
                break;
            }
        }
    }

    // --- §4c trailer-recovery cases ---------------------------------
    //
    // The §4c `AMV_END_` ASCII trailer is the canonical bound on a
    // well-formed file, but field-collected `.amv` files from cheap
    // portable players routinely arrive truncated — the user yanked
    // the SD card, the battery died mid-write, the USB transfer was
    // interrupted. These tests pin the demuxer's graceful-EOF
    // recovery behaviour: any short read at a chunk boundary or
    // inside a chunk body sets `is_truncated()` and returns
    // [`Error::Eof`] for the next call, instead of surfacing a raw
    // I/O error. The bytes that DID parse are still emitted as
    // normal packets.

    /// Happy-path baseline: a complete synthetic file ends via the
    /// `AMV_END_` literal, and `is_truncated()` stays `false`.
    #[test]
    fn complete_file_reports_is_truncated_false() {
        let buf = build_synthetic_file(3);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        assert!(!d.is_truncated(), "fresh demuxer is not truncated");
        loop {
            match d.next_packet() {
                Ok(_) => {}
                Err(Error::Eof) => break,
                Err(e) => panic!("walk error: {e:?}"),
            }
        }
        assert!(
            !d.is_truncated(),
            "drained via AMV_END_ — must not be flagged truncated"
        );
    }

    // --- §4c trailer-offset cross-check -----------------------------
    //
    // The §4c `AMV_END_` trailer is the only byte-bound the stream
    // carries (§1 quirks: zeroed RIFF/LIST sizes, no idx1). The §4
    // no-padding proof turns on its exact file offset — a clean
    // `8 + size` walk must land it at `stream_len - 8`. These tests pin
    // the `trailer_offset()` / `trailer_matches_eof()` accessors that
    // surface that observation.

    /// Before any walk the trailer offset is unknown; after a clean
    /// `AMV_END_`-bounded drain it equals `stream_len - 8` (the trailer
    /// is the final 8 bytes), and `trailer_matches_eof` confirms the
    /// §4 no-padding landing.
    #[test]
    fn trailer_offset_records_clean_eof_position() {
        let buf = build_synthetic_file(3);
        let stream_len = buf.len() as u64;
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        // Not yet observed.
        assert_eq!(d.trailer_offset(), None);
        assert_eq!(d.trailer_matches_eof(stream_len), None);
        loop {
            match d.next_packet() {
                Ok(_) => {}
                Err(Error::Eof) => break,
                Err(e) => panic!("walk error: {e:?}"),
            }
        }
        // The trailer occupies the last 8 bytes of the file.
        assert_eq!(d.trailer_offset(), Some(stream_len - 8));
        assert_eq!(d.trailer_matches_eof(stream_len), Some(true));
        // A stream_len that disagrees with the walk fails the check.
        assert_eq!(d.trailer_matches_eof(stream_len + 1), Some(false));
    }

    /// A truncated stream (no `AMV_END_`) never records a trailer
    /// offset — the three terminal states stay distinguishable:
    /// truncated EOF has `trailer_offset() == None && is_truncated()`.
    #[test]
    fn trailer_offset_stays_none_on_truncation() {
        let mut buf = build_synthetic_file(3);
        buf.truncate(buf.len() - 8); // drop AMV_END_
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        while d.next_packet().is_ok() {}
        assert!(d.is_truncated());
        assert_eq!(d.trailer_offset(), None);
        // With no observed trailer there is nothing to cross-check.
        assert_eq!(d.trailer_matches_eof(1_000), None);
    }

    /// `seek_to` that walks off the end onto the trailer records the
    /// offset too (the linear-seek path shares the §4c terminal check).
    #[test]
    fn trailer_offset_recorded_by_seek_to_past_end() {
        let buf = build_synthetic_file(2);
        let stream_len = buf.len() as u64;
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        let _ = d.seek_to(0, 1_000_000).expect("seek past end");
        assert_eq!(d.trailer_offset(), Some(stream_len - 8));
        assert_eq!(d.trailer_matches_eof(stream_len), Some(true));
    }

    /// Truncation pattern #1: the file ends exactly after the last
    /// complete chunk, with the §4c `AMV_END_` 8-byte trailer
    /// missing entirely. Real-world: writer crashed in the post-
    /// payload finalisation step. The walker emits every complete
    /// packet, then EOFs gracefully with `is_truncated() == true`.
    #[test]
    fn truncated_no_trailer_drains_then_graceful_eof() {
        let mut buf = build_synthetic_file(3);
        // Strip the 8-byte `AMV_END_` trailer the helper appended.
        assert_eq!(&buf[buf.len() - 8..], &AMV_END_TRAILER);
        buf.truncate(buf.len() - 8);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        let mut got = 0;
        loop {
            match d.next_packet() {
                Ok(_) => got += 1,
                Err(Error::Eof) => break,
                Err(e) => panic!("walk error: {e:?}"),
            }
        }
        // 3 video + 3 audio chunks all complete.
        assert_eq!(got, 6, "every complete chunk must still be emitted");
        assert!(
            d.is_truncated(),
            "missing AMV_END_ must flag the stream as truncated"
        );
        // Subsequent calls keep returning EOF and the flag stays set.
        assert!(matches!(d.next_packet().unwrap_err(), Error::Eof));
        assert!(d.is_truncated());
    }

    /// Truncation pattern #2: the file is cut mid-way through the
    /// 8-byte chunk header itself (writer crashed having committed
    /// only 1–7 bytes of the next header). The walker recognises
    /// the short read at the chunk-header boundary and EOFs
    /// gracefully.
    #[test]
    fn truncated_mid_chunk_header_drains_then_graceful_eof() {
        // 4 complete pairs (V+A * 4) then a partial chunk-header of
        // just 3 bytes. We strip the trailer the helper appends
        // before adding the partial header.
        let mut buf = build_synthetic_file(4);
        buf.truncate(buf.len() - 8); // drop AMV_END_
        buf.extend_from_slice(&VIDEO_CHUNK_TAG[..3]); // partial FOURCC
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        let mut got = 0;
        loop {
            match d.next_packet() {
                Ok(_) => got += 1,
                Err(Error::Eof) => break,
                Err(e) => panic!("walk error: {e:?}"),
            }
        }
        assert_eq!(got, 8, "all 4 V+A pairs (8 chunks) must be emitted");
        assert!(d.is_truncated());
    }

    /// Truncation pattern #3: the chunk **header** parses cleanly but
    /// the body falls off the end of the file (writer crashed mid-
    /// payload — common when a 1633-byte JPEG body was only half-
    /// flushed). The walker drops the partial frame and EOFs
    /// gracefully.
    #[test]
    fn truncated_mid_video_body_drains_then_graceful_eof() {
        // 2 complete pairs (V+A * 2), then a video header announcing
        // 1000 bytes but only 50 bytes follow.
        let mut buf = build_synthetic_file(2);
        buf.truncate(buf.len() - 8); // drop AMV_END_
        buf.extend_from_slice(&VIDEO_CHUNK_TAG);
        buf.extend_from_slice(&1000u32.to_le_bytes()); // declares 1000 bytes
        buf.extend_from_slice(&[0xCDu8; 50]); // only 50 actually present
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        let mut got = 0;
        loop {
            match d.next_packet() {
                Ok(_) => got += 1,
                Err(Error::Eof) => break,
                Err(e) => panic!("walk error: {e:?}"),
            }
        }
        assert_eq!(got, 4, "only the 2 complete (V+A) pairs are emitted");
        assert!(d.is_truncated());
    }

    /// Truncation pattern #4: same as #3 but for the audio side
    /// (last `01wb` body truncated). Mirrors the video case so both
    /// `read_*_packet` body-short-read paths are exercised.
    #[test]
    fn truncated_mid_audio_body_drains_then_graceful_eof() {
        let mut buf = build_synthetic_file(2);
        buf.truncate(buf.len() - 8); // drop AMV_END_
                                     // Emit one more complete video chunk so the truncated last
                                     // chunk is an audio chunk (matching the §4 V-then-A
                                     // alternation pattern).
        let extra_video_body = 42u32.to_le_bytes().to_vec();
        buf.extend_from_slice(&VIDEO_CHUNK_TAG);
        buf.extend_from_slice(&(extra_video_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(&extra_video_body);
        // Audio header announces 200 bytes, only 5 follow.
        buf.extend_from_slice(&AUDIO_CHUNK_TAG);
        buf.extend_from_slice(&200u32.to_le_bytes());
        buf.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD, 0xEE]);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        let mut got = 0;
        loop {
            match d.next_packet() {
                Ok(_) => got += 1,
                Err(Error::Eof) => break,
                Err(e) => panic!("walk error: {e:?}"),
            }
        }
        // 2 complete (V,A) pairs + 1 extra complete video chunk = 5.
        assert_eq!(got, 5);
        assert!(d.is_truncated());
    }

    /// Truncation pattern #5: the file is cut after the very last
    /// payload byte, with **zero** bytes following the final
    /// complete chunk. The next chunk-header `read_exact` returns
    /// the no-data short read directly — must EOF gracefully and
    /// be flagged truncated.
    #[test]
    fn truncated_zero_bytes_after_last_chunk_drains_then_graceful_eof() {
        let mut buf = build_synthetic_file(1);
        buf.truncate(buf.len() - 8); // drop the AMV_END_ trailer
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        let _v = d.next_packet().expect("v0");
        let _a = d.next_packet().expect("a0");
        // No more bytes — should be a clean truncated-EOF.
        assert!(matches!(d.next_packet().unwrap_err(), Error::Eof));
        assert!(d.is_truncated());
    }

    /// `is_truncated()` must remain `false` while there are still
    /// chunks left to emit — the flag only flips on the actual
    /// truncation-driven EOF, not preemptively at open() time.
    #[test]
    fn is_truncated_only_flips_at_truncating_eof() {
        let mut buf = build_synthetic_file(3);
        buf.truncate(buf.len() - 8);
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        assert!(!d.is_truncated(), "before any next_packet");
        let _v0 = d.next_packet().expect("v0");
        assert!(!d.is_truncated(), "mid-walk, before EOF");
        // Drain the rest.
        while d.next_packet().is_ok() {}
        assert!(d.is_truncated(), "after the truncating EOF");
    }

    /// `build_chunk_index` must apply the same recovery semantics as
    /// the live walker: a truncated tail breaks the build cleanly
    /// (no error surface), the index covers every chunk that did
    /// land, and the walker state is preserved.
    #[test]
    fn build_chunk_index_recovers_from_missing_trailer() {
        let mut buf = build_synthetic_file(4);
        buf.truncate(buf.len() - 8); // drop AMV_END_
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        d.build_chunk_index()
            .expect("build_chunk_index must tolerate truncation");
        let idx = d.chunk_index().expect("index populated");
        // 4 video + 4 audio chunks complete.
        assert_eq!(idx.len(), 8);
        // Walker still at start.
        assert_eq!(d.cursor(), 0x13C);
        // is_truncated is a walker concept, not an index concept —
        // the index build itself does not flip it.
        assert!(!d.is_truncated());
    }

    /// `build_chunk_index` on a file truncated mid-audio-preamble
    /// must not error — it stops at the last fully-readable chunk
    /// instead.
    #[test]
    fn build_chunk_index_recovers_from_mid_preamble_truncation() {
        // 2 complete pairs, one extra complete video chunk, then a
        // partial audio chunk: 8-byte header announcing 12 bytes,
        // but only 3 bytes of the 8-byte preamble actually present.
        let mut buf = build_synthetic_file(2);
        buf.truncate(buf.len() - 8); // drop AMV_END_
        let extra_video_body = 42u32.to_le_bytes().to_vec();
        buf.extend_from_slice(&VIDEO_CHUNK_TAG);
        buf.extend_from_slice(&(extra_video_body.len() as u32).to_le_bytes());
        buf.extend_from_slice(&extra_video_body);
        buf.extend_from_slice(&AUDIO_CHUNK_TAG);
        buf.extend_from_slice(&12u32.to_le_bytes());
        buf.extend_from_slice(&[0x00, 0x00, 0x00]); // 3 of 8 preamble bytes
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        d.build_chunk_index()
            .expect("build_chunk_index must tolerate mid-preamble truncation");
        let idx = d.chunk_index().expect("index populated");
        // 2 V + 2 A + 1 extra V = 5 fully indexed chunks. The
        // partial audio chunk is dropped.
        assert_eq!(idx.len(), 5);
    }

    /// After a truncated walk, an indexed seek built post-truncation
    /// must still land correctly on the chunks that did make it.
    #[test]
    fn indexed_seek_after_truncated_build_lands_correctly() {
        let mut buf = build_synthetic_file(5);
        buf.truncate(buf.len() - 8); // drop AMV_END_
        let mut d = AmvDemuxer::open(Cursor::new(buf)).expect("open");
        d.build_chunk_index().expect("build_chunk_index");
        let landed = d.seek_to(0, 3).expect("seek to v3");
        assert_eq!(landed, 3);
        let p = d.next_packet().expect("packet after seek");
        assert_eq!(p.stream_index, 0);
        assert_eq!(p.pts, Some(3));
        assert_eq!(p.data, 3u32.to_le_bytes().to_vec());
    }

    // ─────────── §2/§3 strict-mode entrypoint coverage ───────────

    /// A synthetic comedian-profile file opens cleanly through the new
    /// strict entrypoint and walks identically to the permissive
    /// entrypoint — first packet PTS, frame count, EOF marker all
    /// match.
    #[test]
    fn open_strict_accepts_synthetic_comedian_profile() {
        let buf = build_synthetic_file(2);
        let mut d = AmvDemuxer::open_strict(Cursor::new(buf)).expect("strict open");
        assert_eq!(d.header().width, 128);
        assert_eq!(d.header().height, 96);
        assert_eq!(d.header().fps, 12);
        let v0 = d.next_packet().expect("v0");
        assert_eq!(v0.pts, Some(0));
        let _a0 = d.next_packet().expect("a0");
        let v1 = d.next_packet().expect("v1");
        assert_eq!(v1.pts, Some(1));
        let _a1 = d.next_packet().expect("a1");
        assert!(matches!(d.next_packet().unwrap_err(), Error::Eof));
    }

    /// Strict-mode rejects a corrupted `flag_one` (§2 +0x2C constant)
    /// that the permissive entrypoint silently accepts. This is the
    /// only behavioural difference between the two entrypoints — see
    /// `parse::tests::parse_strict_with_permissive_open_still_accepts_corrupted_input`
    /// for the permissive companion.
    #[test]
    fn open_strict_rejects_corrupted_flag_one() {
        let mut buf = build_synthetic_prelude(128, 96, 12, 0x0000_0121, 22_050);
        // amvh body sits at file 0x20; flag_one is body +0x2C → file 0x4C.
        buf[0x4C] = 0x07;
        // Append a minimal trailer so the permissive path would happily
        // continue (proving the difference is in the sentinel check,
        // not in the body walk).
        buf.extend_from_slice(&AMV_END_TRAILER);
        let err = AmvDemuxer::open_strict(Cursor::new(buf.clone())).unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => assert!(msg.contains("+0x2C")),
            other => panic!("expected InvalidData(flag_one), got {other:?}"),
        }
        // The permissive entrypoint must keep accepting it.
        let _d = AmvDemuxer::open(Cursor::new(buf)).expect("permissive accepts corrupted flag_one");
    }

    /// Strict-mode rejects a synthetic prelude whose video `strh` body
    /// is non-zero (§3 records the body as "all zero"). The permissive
    /// entrypoint accepts it because the permissive path never validates
    /// strh body content.
    #[test]
    fn open_strict_rejects_non_zero_video_strh_body() {
        let mut buf = build_synthetic_prelude(128, 96, 12, 0x0000_0121, 22_050);
        // First byte of the video strh body — well inside the
        // STRH_VIDEO_BODY_LEN==0x38 run. Inject any non-zero byte to
        // trigger the strict-mode rejection.
        //
        // Layout (§3a, also `STRL_VIDEO_OFFSET = 0x58`):
        //   0x58 LIST | 0x5C size=0 | 0x60 'strl' | 0x64 'strh'
        //   0x68 strh-size (4 bytes) | 0x6C strh-body...
        let v_strh_body_start_in_prelude = 0x6C;
        buf[v_strh_body_start_in_prelude] = 0xFE;
        buf.extend_from_slice(&AMV_END_TRAILER);
        let err = AmvDemuxer::open_strict(Cursor::new(buf.clone())).unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => assert!(msg.contains("strh")),
            other => panic!("expected InvalidData(strh), got {other:?}"),
        }
        let _d = AmvDemuxer::open(Cursor::new(buf)).expect("permissive accepts non-zero strh");
    }

    /// Strict-mode rejects a synthetic prelude whose audio `strf`
    /// WAVEFORMATEX violates a §3b sentinel — here `nBlockAlign` is
    /// flipped from 2 to 4. The permissive entrypoint must keep
    /// accepting it because the permissive path never gates on the
    /// audio strf body values.
    #[test]
    fn open_strict_rejects_wrong_audio_block_align() {
        let mut buf = build_synthetic_prelude(128, 96, 12, 0x0000_0121, 22_050);
        // The audio strf body sits at file 0x11C in every synthetic
        // prelude:
        //   0x00 RIFF/0/'AMV '
        //   0x0C LIST/0/'hdrl'
        //   0x18 'amvh' + size 0x38 → body 0x20..0x58
        //   0x58 LIST/0/'strl' (video)
        //   0x64 'strh' + size 0x38 → body 0x6C..0xA4
        //   0xA4 'strf' + size 0x24 → body 0xAC..0xD0
        //   0xD0 LIST/0/'strl' (audio)
        //   0xDC 'strh' + size 0x30 → body 0xE4..0x114
        //   0x114 'strf' + size 0x14 → body 0x11C..0x130
        // `nBlockAlign` is at body +0x0C → file 0x128.
        let block_align_off = 0x11Cusize + 0x0C;
        buf[block_align_off..block_align_off + 2].copy_from_slice(&4u16.to_le_bytes());
        buf.extend_from_slice(&AMV_END_TRAILER);
        let err = AmvDemuxer::open_strict(Cursor::new(buf.clone())).unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => assert!(msg.contains("nBlockAlign")),
            other => panic!("expected InvalidData(nBlockAlign), got {other:?}"),
        }
        let _d = AmvDemuxer::open(Cursor::new(buf))
            .expect("permissive accepts non-canonical nBlockAlign");
    }

    /// Real-fixture cross-check: the staged `comedian.amv` device file
    /// passes the strict §2/§3 sentinel check. This is the load-bearing
    /// "real bytes from a real device satisfy our trace-derived
    /// invariants" assertion that lets the strict entrypoint be trusted
    /// as a "this is a real AMV file" filter.
    #[test]
    fn comedian_fixture_open_strict_succeeds() {
        let crate_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/comedian.amv");
        let workspace_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/container/amv/fixtures/comedian.amv");
        let path = if crate_path.exists() {
            crate_path
        } else if workspace_path.exists() {
            workspace_path
        } else {
            eprintln!("skipping comedian strict-open test: fixture not staged");
            return;
        };
        let f = std::fs::File::open(&path).expect("open fixture");
        let d = AmvDemuxer::open_strict(std::io::BufReader::new(f))
            .expect("strict open accepts real comedian.amv");
        // Sanity check the parsed parameters match §2 + §3b.
        assert_eq!(d.header().width, 128);
        assert_eq!(d.header().height, 96);
        assert_eq!(d.header().fps, 12);
        assert_eq!(d.header().micros_per_frame, 83_333);
        assert_eq!(d.header().flag_one, 1);
        assert_eq!(d.header().reserved_30, 0);
        assert_eq!(d.audio_format().samples_per_sec, 22_050);
        // §3b WAVEFORMATEX device-profile constants — the new sentinel
        // suite gates on these directly inside open_strict.
        assert_eq!(d.audio_format().format_tag, 1);
        assert_eq!(d.audio_format().channels, 1);
        assert_eq!(d.audio_format().avg_bytes_per_sec, 44_100);
        assert_eq!(d.audio_format().block_align, 2);
        assert_eq!(d.audio_format().bits_per_sample, 16);
        assert_eq!(d.audio_format().cb_size, 0);
    }

    /// Real-fixture cross-check: walk the staged comedian.amv to EOF
    /// and run the §4a video-payload shape validator on **every** video
    /// chunk plus the §4b audio-preamble sentinel on every audio chunk.
    ///
    /// All 1116 video frames are expected to start with `FF D8` (SOI)
    /// and end with `FF D9` (EOI) per the trace doc's "self-contained
    /// JPEG bracketed by SOI..EOI" observation, and all 1116 audio
    /// blocks are expected to carry a positive `decoded_sample_count`.
    /// Any failure surfaces the offending chunk index with the validator
    /// error so a regression points directly at the contaminated chunk.
    #[test]
    fn comedian_fixture_all_chunks_pass_payload_shape_sentinels() {
        use crate::parse::{validate_video_payload_shape, AmvAudioPreamble};

        let crate_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/comedian.amv");
        let workspace_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/container/amv/fixtures/comedian.amv");
        let path = if crate_path.exists() {
            crate_path
        } else if workspace_path.exists() {
            workspace_path
        } else {
            eprintln!("skipping comedian payload-shape sentinel test: fixture not staged");
            return;
        };
        let f = std::fs::File::open(&path).expect("open fixture");
        let mut d = AmvDemuxer::open(std::io::BufReader::new(f)).expect("open comedian.amv");

        let mut video_chunks_validated = 0u32;
        let mut audio_chunks_validated = 0u32;
        loop {
            match d.next_packet() {
                Ok(p) if p.stream_index == 0 => {
                    validate_video_payload_shape(&p.data).unwrap_or_else(|e| {
                        panic!(
                            "video chunk #{video_chunks_validated} failed §4a shape check: {e:?}"
                        )
                    });
                    video_chunks_validated += 1;
                }
                Ok(p) if p.stream_index == 1 => {
                    let preamble = AmvAudioPreamble::parse(&p.data).unwrap_or_else(|e| {
                        panic!("audio chunk #{audio_chunks_validated} preamble parse: {e:?}")
                    });
                    preamble.validate_sentinels().unwrap_or_else(|e| {
                        panic!(
                            "audio chunk #{audio_chunks_validated} failed §4b sentinel check: {e:?}"
                        )
                    });
                    audio_chunks_validated += 1;
                }
                Ok(_) => panic!("unexpected stream index"),
                Err(Error::Eof) => break,
                Err(other) => panic!("walk error: {other:?}"),
            }
        }
        assert_eq!(
            video_chunks_validated, 1116,
            "expected to validate 1116 video chunks"
        );
        assert_eq!(
            audio_chunks_validated, 1116,
            "expected to validate 1116 audio chunks"
        );
    }

    /// Real-fixture cross-check for the §4a strict-marker invariant:
    /// walk the staged `comedian.amv` to EOF and run
    /// `validate_video_payload_no_internal_markers` on every video
    /// chunk. Per the trace doc the device-stripped bitstream carries
    /// **no** internal JPEG marker segments — every one of the 1116
    /// video frames must pass the strict-marker scan, since the trace
    /// records the player splices its hardcoded quant / Huffman tables
    /// back in before decoding and the on-disk bytes between SOI and
    /// EOI are pure entropy-coded data (with at most `FF 00` byte
    /// stuffing).
    #[test]
    fn comedian_fixture_all_video_chunks_pass_no_internal_markers() {
        use crate::parse::validate_video_payload_no_internal_markers;

        let crate_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/comedian.amv");
        let workspace_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/container/amv/fixtures/comedian.amv");
        let path = if crate_path.exists() {
            crate_path
        } else if workspace_path.exists() {
            workspace_path
        } else {
            eprintln!("skipping comedian strict-marker sentinel test: fixture not staged");
            return;
        };
        let f = std::fs::File::open(&path).expect("open fixture");
        let mut d = AmvDemuxer::open(std::io::BufReader::new(f)).expect("open comedian.amv");

        let mut video_chunks_validated = 0u32;
        loop {
            match d.next_packet() {
                Ok(p) if p.stream_index == 0 => {
                    validate_video_payload_no_internal_markers(&p.data).unwrap_or_else(|e| {
                        panic!(
                            "video chunk #{video_chunks_validated} failed §4a \
                             strict-marker check: {e:?}"
                        )
                    });
                    video_chunks_validated += 1;
                }
                Ok(_) => {}
                Err(Error::Eof) => break,
                Err(other) => panic!("walk error: {other:?}"),
            }
        }
        assert_eq!(
            video_chunks_validated, 1116,
            "expected to strict-marker-validate 1116 video chunks"
        );
    }

    /// Real-fixture cross-check for [`MoviPayloadIter`]: load
    /// `comedian.amv` into memory, locate the `movi` payload bytes
    /// (between the `movi` FOURCC opener and the §4c 8-byte
    /// `AMV_END_` trailer), and confirm the typed iterator walks
    /// the same 1116 + 1116 = 2232 chunks the demuxer's I/O path
    /// surfaces — with strict §4 video-first alternation and every
    /// `01wb` block's preamble carrying `decoded_sample_count = 1837`
    /// (the §4b worked-example value `22_050 ÷ 12`).
    #[test]
    fn comedian_fixture_movi_payload_iter_walks_2232_chunks() {
        use crate::parse::{
            validate_movi_interleave, MoviPayload, MoviPayloadIter, AMV_END_TRAILER,
        };

        let crate_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/comedian.amv");
        let workspace_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/container/amv/fixtures/comedian.amv");
        let path = if crate_path.exists() {
            crate_path
        } else if workspace_path.exists() {
            workspace_path
        } else {
            eprintln!("skipping comedian MoviPayloadIter fixture test: not staged");
            return;
        };
        let bytes = std::fs::read(&path).expect("read fixture");

        // Locate the `movi` FOURCC inside the file; the §4 payload
        // begins on the byte *after* it.
        let movi_pos = bytes
            .windows(4)
            .position(|w| w == b"movi")
            .expect("movi FOURCC present");
        let movi_body_start = movi_pos + 4;
        // §4c: the file ends in an 8-byte `AMV_END_` literal — slice
        // it off so the iterator walks the chunk run only.
        assert!(bytes.len() >= AMV_END_TRAILER.len());
        let trailer_start = bytes.len() - AMV_END_TRAILER.len();
        assert_eq!(&bytes[trailer_start..], &AMV_END_TRAILER);
        let movi_body = &bytes[movi_body_start..trailer_start];

        let mut iter = MoviPayloadIter::new(movi_body);
        let mut kinds = Vec::with_capacity(2232);
        let mut n_video = 0u32;
        let mut n_audio = 0u32;
        let mut first_video_size = None;
        let mut first_audio_decoded = None;
        for item in iter.by_ref() {
            let payload = item.expect("comedian.amv movi walk must not error");
            match payload {
                MoviPayload::Video { body, .. } => {
                    if first_video_size.is_none() {
                        first_video_size = Some(body.len());
                    }
                    n_video += 1;
                    kinds.push(crate::parse::ChunkKind::Video);
                }
                MoviPayload::Audio { preamble, .. } => {
                    if first_audio_decoded.is_none() {
                        first_audio_decoded = Some(preamble.decoded_sample_count);
                    }
                    n_audio += 1;
                    kinds.push(crate::parse::ChunkKind::Audio);
                }
                MoviPayload::Other { tag, .. } => {
                    panic!(
                        "comedian.amv unexpected non-00dc/01wb chunk: \
                         {:02X} {:02X} {:02X} {:02X}",
                        tag[0], tag[1], tag[2], tag[3]
                    );
                }
            }
        }
        // §4 worked-example invariants.
        assert_eq!(n_video, 1116, "expected 1116 video chunks");
        assert_eq!(n_audio, 1116, "expected 1116 audio chunks");
        assert_eq!(first_video_size, Some(1633), "first video size = 0x661");
        assert_eq!(first_audio_decoded, Some(1837), "first audio = 22050/12");
        // Cursor must land cleanly on the end of the movi body slice
        // (i.e. 8 bytes before EOF in the original file).
        assert_eq!(iter.cursor(), movi_body.len());
        // Strict §4 alternation must hold across the full walk.
        validate_movi_interleave(&kinds).expect("comedian.amv chunk run satisfies §4");
    }

    /// Real-fixture cross-check for the §4b nibble-packing budget: walk
    /// `comedian.amv`'s `movi` payload, parse every `01wb` block's
    /// preamble, and confirm the block-length cross-check
    /// `AmvAudioPreamble::is_consistent_with_body_len`.
    ///
    /// The trace's §4b worked example is exact for the **first** audio
    /// block: `1837` decoded samples in a `927`-byte payload (`8`
    /// preamble + `919 = ceil(1837 / 2)` nibble body). The trace also
    /// records that the block size is *near*-constant — "927 bytes,
    /// occasionally 930" — so not every block satisfies the exact
    /// `body == ceil(samples / 2)` relation; some carry a few padding
    /// bytes. This test therefore pins the first block exactly (the
    /// documented worked example) and asserts that the **majority** of
    /// blocks satisfy the nibble budget, surfacing the matched / total
    /// split without over-asserting on the padded outliers the trace
    /// explicitly notes.
    #[test]
    fn comedian_fixture_audio_blocks_nibble_budget() {
        use crate::parse::{MoviPayload, MoviPayloadIter, AMV_END_TRAILER};

        let crate_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/comedian.amv");
        let workspace_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/container/amv/fixtures/comedian.amv");
        let path = if crate_path.exists() {
            crate_path
        } else if workspace_path.exists() {
            workspace_path
        } else {
            eprintln!("skipping comedian nibble-budget fixture test: not staged");
            return;
        };
        let bytes = std::fs::read(&path).expect("read fixture");

        let movi_pos = bytes
            .windows(4)
            .position(|w| w == b"movi")
            .expect("movi FOURCC present");
        let movi_body_start = movi_pos + 4;
        let trailer_start = bytes.len() - AMV_END_TRAILER.len();
        assert_eq!(&bytes[trailer_start..], &AMV_END_TRAILER);
        let movi_body = &bytes[movi_body_start..trailer_start];

        let mut total_audio = 0u32;
        let mut budget_matches = 0u32;
        let mut first_audio_checked = false;
        for item in MoviPayloadIter::new(movi_body) {
            let payload = item.expect("comedian.amv movi walk must not error");
            if let MoviPayload::Audio { preamble, body, .. } = payload {
                total_audio += 1;
                let total_len = body.len() as u64;
                if !first_audio_checked {
                    // §4b worked example: first block is exactly
                    // 1837 samples in a 927-byte payload (8 + 919).
                    assert_eq!(preamble.decoded_sample_count, 1837);
                    assert_eq!(total_len, 927);
                    assert_eq!(preamble.nibble_body_len(), 919);
                    assert!(
                        preamble.is_consistent_with_body_len(total_len),
                        "first audio block must satisfy the exact §4b nibble budget"
                    );
                    first_audio_checked = true;
                }
                if preamble.is_consistent_with_body_len(total_len) {
                    budget_matches += 1;
                }
            }
        }

        assert_eq!(total_audio, 1116, "expected 1116 audio chunks");
        assert!(
            first_audio_checked,
            "must have seen at least one audio block"
        );
        // The trace records "927 bytes, occasionally 930" — the exact
        // nibble relation holds for the common case, so the matched count
        // must dominate (well over half) even though a few padded blocks
        // miss it. This is a loose, trace-faithful lower bound, not an
        // all-must-match assertion.
        assert!(
            budget_matches * 2 > total_audio,
            "expected the majority of audio blocks to satisfy the §4b \
             nibble budget, got {budget_matches} / {total_audio}"
        );
    }
}
