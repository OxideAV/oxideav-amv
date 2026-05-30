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
    CodecId, CodecParameters, Demuxer, Error, MediaType, Packet, PixelFormat, Rational, ReadSeek,
    Result, StreamInfo, TimeBase,
};

use crate::parse::{
    AmvHeader, AmvPrelude, AmvWaveFormat, ChunkHeader, ChunkKind, AMV_END_TRAILER, PRELUDE_MIN_LEN,
};
use crate::{AUDIO_CODEC_ID, VIDEO_CODEC_ID};

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

/// The pure-Rust AMV demuxer. Owns a seekable byte reader (anything
/// implementing [`ReadSeek`]) and walks the AMV chunk tree as
/// described in the workspace trace document.
pub struct AmvDemuxer {
    reader: Box<dyn ReadSeek>,
    streams: [StreamInfo; 2],
    header: AmvHeader,
    audio_format: AmvWaveFormat,
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
}

impl std::fmt::Debug for AmvDemuxer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmvDemuxer")
            .field("header", &self.header)
            .field("audio_format", &self.audio_format)
            .field("cursor", &self.cursor)
            .field("eof", &self.eof)
            .field("next_video_pts", &self.next_video_pts)
            .field("next_audio_pts", &self.next_audio_pts)
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
        let mut reader: Box<dyn ReadSeek> = Box::new(reader);
        reader.seek(SeekFrom::Start(0))?;
        let mut prelude = vec![0u8; PRELUDE_MIN_LEN];
        reader.read_exact(&mut prelude)?;
        let parsed = AmvPrelude::parse(&prelude)?;

        let header = parsed.header;
        let audio_format = parsed.audio_format;
        let movi_start = parsed.movi_payload_start;

        // Video stream: 1/fps clock, MJPEG codec id, width/height
        // from amvh. Frame-rate exposed as a Rational so re-muxers
        // (or PTS-aware codec wrappers) can recover the exact cadence
        // without rounding through micros_per_frame.
        let mut video_params = CodecParameters::video(CodecId::new(VIDEO_CODEC_ID));
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
            cursor: movi_start,
            eof: false,
            next_video_pts: 0,
            next_audio_pts: 0,
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

    /// Convenience: current `movi`-walk cursor (file offset of the
    /// next chunk header). Exposed for tests + audit; not part of the
    /// stable contract.
    #[cfg(test)]
    pub(crate) fn cursor(&self) -> u64 {
        self.cursor
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
        if let Err(e) = self.reader.read_exact(&mut header_bytes) {
            // A short read at this point means the file ended before
            // the trailer landed. Treat as EOF rather than corrupt.
            self.eof = true;
            return Err(Error::other(format!(
                "amv: short read at chunk header: {e}"
            )));
        }
        // First check the trailer literal — its 8 bytes match no
        // valid chunk-header layout (the would-be size field is
        // ASCII `b"END_"` = 0x5F444E45, which is enormous and not
        // sane; checking the literal first is simpler).
        if header_bytes == AMV_END_TRAILER {
            self.eof = true;
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
}

impl AmvDemuxer {
    fn read_video_packet(&mut self, chunk: ChunkHeader) -> Result<Packet> {
        let mut data = vec![0u8; chunk.size as usize];
        self.reader
            .read_exact(&mut data)
            .map_err(|e| Error::other(format!("amv: video chunk body short read: {e}")))?;
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
        self.reader
            .read_exact(&mut data)
            .map_err(|e| Error::other(format!("amv: audio chunk body short read: {e}")))?;
        self.cursor += chunk.advance_total();
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
        let mut buf = build_synthetic_prelude(128, 96, 12, 0x0000_0121, 22_050);
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
        assert_eq!(d.cursor(), 0x13C);
        assert_eq!(d.duration_micros(), Some(93_000_000));
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
    }
}
