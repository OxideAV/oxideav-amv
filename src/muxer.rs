//! AMV muxer — `oxideav_core::Muxer` trait implementation.
//!
//! Writes a byte-exact AMV file conforming to the layout documented in
//! `docs/container/amv/amv-container-trace.md`. The muxer is the inverse
//! of the demuxer in this same crate; the round-trip (mux → demux) is
//! covered by the test suite.
//!
//! ## What this writer emits
//!
//! 1. The full prelude (offsets 0 .. 0x13C) with the exact FOURCCs,
//!    zeroed RIFF / LIST sizes, fully populated `amvh` body, the two
//!    all-zero `strh` / `strf` bodies on the video side, the all-zero
//!    audio `strh`, and the 20-byte `WAVEFORMATEX` on the audio side
//!    (§1, §2, §3).
//! 2. One leaf chunk per packet inside `movi`. Stream index 0 → `00dc`,
//!    stream index 1 → `01wb`. Chunks are NOT padded to an even byte
//!    boundary — the cursor advances by exactly `8 + size` per chunk,
//!    matching §4 "no-padding rule".
//! 3. The 8-byte `AMV_END_` ASCII trailer immediately after the last
//!    packet (§4c). No `idx1` index is written; AMV does not carry one.
//!
//! ## Inputs the writer needs
//!
//! From the supplied `&[StreamInfo]`:
//! - stream 0 must be video with `params.width`, `params.height`,
//!   `params.frame_rate` populated;
//! - stream 1 must be audio with `params.sample_rate` and
//!   `params.channels` populated.
//!
//! The packed-byte duration (`amvh +0x34`) is initialised to zero in
//! `write_header` and patched in `write_trailer` from the observed
//! video-frame count divided by `fps` — that matches what real device
//! files carry (see worked example in `docs/container/amv/`: 1116 frames
//! ÷ 12 fps = 93 s = 1:33).

use std::io::{Seek, SeekFrom, Write};

use oxideav_core::{Error, Muxer, Packet, Result, StreamInfo, WriteSeek};

use crate::parse::{
    AMVH_BODY_LEN, AMV_END_TRAILER, AMV_FORM_TYPE, AUDIO_CHUNK_TAG, VIDEO_CHUNK_TAG,
};

/// File offset where the `amvh +0x34` packed-duration dword lives. Used
/// by [`AmvMuxer::write_trailer`] to patch in the final duration.
/// `0x18` (amvh FOURCC) + 8 (FOURCC+len) + 0x34 = 0x54.
const AMVH_DURATION_FILE_OFFSET: u64 = 0x54;

/// Constants that mirror `parse.rs` private items. The writer needs
/// these byte-lengths verbatim so the prelude offsets line up; we keep
/// our own copies (and have a test that pins them to the demuxer's
/// constants) rather than re-exporting private items from `parse.rs`.
const STRH_VIDEO_BODY_LEN: u32 = 0x38;
const STRF_VIDEO_BODY_LEN: u32 = 0x24;
const STRH_AUDIO_BODY_LEN: u32 = 0x30;
const STRF_AUDIO_BODY_LEN: u32 = 0x14;

/// Stream indices the muxer expects in its `streams` argument.
const STREAM_INDEX_VIDEO: u32 = 0;
const STREAM_INDEX_AUDIO: u32 = 1;

/// Pure-Rust AMV muxer. Constructed via [`AmvMuxer::open`] (or through
/// `oxideav_core::ContainerRegistry::open_muxer("amv", …)` once the
/// container is registered via [`crate::register`]).
pub struct AmvMuxer {
    writer: Box<dyn WriteSeek>,
    /// Resolution declared in the `amvh` body. Lifted from the video
    /// stream's `params.width` / `params.height`.
    width: u32,
    height: u32,
    /// Frames-per-second declared in `amvh`. Lifted from the video
    /// stream's `params.frame_rate` (numerator over denominator → integer
    /// floor; AMV's `amvh` carries an integer fps only).
    fps: u32,
    /// Sample rate / channel count declared in the audio `strf`.
    samples_per_sec: u32,
    channels: u16,
    /// Frame count accumulated across `write_packet` calls on stream 0.
    /// Used by [`Muxer::write_trailer`] to patch the `amvh` packed
    /// duration (§2 worked example: 1116 ÷ 12 = 93 s).
    video_frame_count: u64,
    /// `true` once `write_header` has run successfully. `write_packet`
    /// errors out if called before the header is written.
    header_written: bool,
}

impl std::fmt::Debug for AmvMuxer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmvMuxer")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("fps", &self.fps)
            .field("samples_per_sec", &self.samples_per_sec)
            .field("channels", &self.channels)
            .field("video_frame_count", &self.video_frame_count)
            .field("header_written", &self.header_written)
            .finish()
    }
}

impl AmvMuxer {
    /// Build an `AmvMuxer` from a seekable writer + the two streams
    /// (video first, audio second). Returns `Error::InvalidData` if the
    /// stream layout does not match AMV's two-stream contract.
    pub fn open(writer: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Self> {
        if streams.len() != 2 {
            return Err(Error::invalid(format!(
                "amv: muxer requires exactly 2 streams (video then audio), got {}",
                streams.len()
            )));
        }
        let video = &streams[0];
        let audio = &streams[1];
        if video.index != STREAM_INDEX_VIDEO {
            return Err(Error::invalid(format!(
                "amv: stream[0] must have index {STREAM_INDEX_VIDEO} (video), got {}",
                video.index
            )));
        }
        if audio.index != STREAM_INDEX_AUDIO {
            return Err(Error::invalid(format!(
                "amv: stream[1] must have index {STREAM_INDEX_AUDIO} (audio), got {}",
                audio.index
            )));
        }
        let width = video
            .params
            .width
            .ok_or_else(|| Error::invalid("amv: video stream missing params.width".to_string()))?;
        let height = video
            .params
            .height
            .ok_or_else(|| Error::invalid("amv: video stream missing params.height".to_string()))?;
        // frame_rate is a Rational; AMV's `amvh` only carries an integer
        // fps so we use the floor of num/den. The trace examples (12 and
        // 16 fps) are both exact integers; non-integer cadences are not
        // representable in AMV's `amvh` body.
        let fr = video.params.frame_rate.ok_or_else(|| {
            Error::invalid("amv: video stream missing params.frame_rate".to_string())
        })?;
        if fr.num <= 0 || fr.den <= 0 {
            return Err(Error::invalid(format!(
                "amv: video frame_rate must be positive, got {}/{}",
                fr.num, fr.den
            )));
        }
        let fps_i = fr.num / fr.den;
        if fps_i <= 0 || fps_i > u32::MAX as i64 {
            return Err(Error::invalid(format!(
                "amv: video fps out of range: {}/{} = {fps_i}",
                fr.num, fr.den
            )));
        }
        let fps = fps_i as u32;

        let samples_per_sec = audio.params.sample_rate.ok_or_else(|| {
            Error::invalid("amv: audio stream missing params.sample_rate".to_string())
        })?;
        let channels = audio.params.channels.ok_or_else(|| {
            Error::invalid("amv: audio stream missing params.channels".to_string())
        })?;

        Ok(Self {
            writer,
            width,
            height,
            fps,
            samples_per_sec,
            channels,
            video_frame_count: 0,
            header_written: false,
        })
    }
}

impl Muxer for AmvMuxer {
    fn format_name(&self) -> &str {
        crate::CONTAINER_NAME
    }

    fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Err(Error::invalid("amv: write_header called twice".to_string()));
        }
        let prelude = build_prelude_bytes(
            self.width,
            self.height,
            self.fps,
            0, /* packed duration patched in write_trailer */
            self.samples_per_sec,
            self.channels,
        );
        self.writer
            .seek(SeekFrom::Start(0))
            .map_err(|e| Error::other(format!("amv: seek to start: {e}")))?;
        self.writer
            .write_all(&prelude)
            .map_err(|e| Error::other(format!("amv: write prelude: {e}")))?;
        self.header_written = true;
        Ok(())
    }

    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::invalid(
                "amv: write_packet called before write_header".to_string(),
            ));
        }
        if packet.data.len() > u32::MAX as usize {
            return Err(Error::invalid(format!(
                "amv: chunk size exceeds 4 GiB: {}",
                packet.data.len()
            )));
        }
        let tag = match packet.stream_index {
            STREAM_INDEX_VIDEO => {
                self.video_frame_count = self.video_frame_count.saturating_add(1);
                VIDEO_CHUNK_TAG
            }
            STREAM_INDEX_AUDIO => AUDIO_CHUNK_TAG,
            other => {
                return Err(Error::invalid(format!(
                    "amv: stream_index must be 0 (video) or 1 (audio), got {other}"
                )));
            }
        };
        // Write the 8-byte leaf-chunk header followed by the payload.
        // Per §4 "no-padding rule" we never pad to an even byte.
        self.writer
            .write_all(&tag)
            .map_err(|e| Error::other(format!("amv: write chunk tag: {e}")))?;
        let size = packet.data.len() as u32;
        self.writer
            .write_all(&size.to_le_bytes())
            .map_err(|e| Error::other(format!("amv: write chunk size: {e}")))?;
        self.writer
            .write_all(&packet.data)
            .map_err(|e| Error::other(format!("amv: write chunk body: {e}")))?;
        Ok(())
    }

    fn write_trailer(&mut self) -> Result<()> {
        if !self.header_written {
            return Err(Error::invalid(
                "amv: write_trailer called before write_header".to_string(),
            ));
        }
        // Append the AMV_END_ 8-byte trailer per §4c.
        self.writer
            .write_all(&AMV_END_TRAILER)
            .map_err(|e| Error::other(format!("amv: write trailer: {e}")))?;
        // Patch the `amvh` packed duration from the accumulated video
        // frame count (§2 worked example: 1116 frames ÷ 12 fps = 93 s
        // → 0x21 0x01 0x00 0x00). If `fps` is zero something earlier
        // would have rejected the open; guard anyway to avoid division
        // by zero.
        if self.fps > 0 && self.video_frame_count > 0 {
            let total_seconds = self.video_frame_count / self.fps as u64;
            let hours = (total_seconds / 3600).min(u8::MAX as u64) as u8;
            let minutes = ((total_seconds % 3600) / 60) as u8;
            let seconds = (total_seconds % 60) as u8;
            let packed = pack_duration(seconds, minutes, hours);
            self.writer
                .seek(SeekFrom::Start(AMVH_DURATION_FILE_OFFSET))
                .map_err(|e| Error::other(format!("amv: seek to amvh duration: {e}")))?;
            self.writer
                .write_all(&packed.to_le_bytes())
                .map_err(|e| Error::other(format!("amv: patch amvh duration: {e}")))?;
            // Leave the cursor past the trailer — re-seek to end of
            // file so further writes (if any) don't clobber the
            // payload. The Muxer trait does not promise a cursor
            // location after `write_trailer`, but it's polite.
            self.writer
                .seek(SeekFrom::End(0))
                .map_err(|e| Error::other(format!("amv: seek to end: {e}")))?;
        }
        Ok(())
    }
}

/// Build the 0x13C-byte prelude verbatim per §1..§3 of the trace doc.
/// `duration_packed` is the raw little-endian u32 to write at amvh
/// +0x34. Pass `0` when assembling the header up-front; the muxer
/// patches the real value in `write_trailer`.
pub(crate) fn build_prelude_bytes(
    width: u32,
    height: u32,
    fps: u32,
    duration_packed: u32,
    samples_per_sec: u32,
    channels: u16,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(0x140);
    // ── §1 Top-level RIFF + FORM ──────────────────────────────────
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&[0u8; 4]); // RIFF size — zeroed per §1 quirk #1.
    buf.extend_from_slice(&AMV_FORM_TYPE);
    // hdrl LIST opener.
    buf.extend_from_slice(b"LIST");
    buf.extend_from_slice(&[0u8; 4]); // LIST size — zeroed per §1 quirk #1.
    buf.extend_from_slice(b"hdrl");
    // ── §2 amvh leaf ──────────────────────────────────────────────
    buf.extend_from_slice(b"amvh");
    buf.extend_from_slice(&AMVH_BODY_LEN.to_le_bytes());
    let mut amvh_body = vec![0u8; AMVH_BODY_LEN as usize];
    // dwMicroSecPerFrame at +0x00 — 1_000_000 / fps. `checked_div`
    // guards the synthetic-fps-0 case used by some helper-side tests
    // without taking an unnecessary panic branch.
    let micros = 1_000_000u32.checked_div(fps).unwrap_or(0);
    amvh_body[0x00..0x04].copy_from_slice(&micros.to_le_bytes());
    // width at +0x20, height at +0x24, fps at +0x28.
    amvh_body[0x20..0x24].copy_from_slice(&width.to_le_bytes());
    amvh_body[0x24..0x28].copy_from_slice(&height.to_le_bytes());
    amvh_body[0x28..0x2C].copy_from_slice(&fps.to_le_bytes());
    // Constant `1` flag at +0x2C — matches both observed fixtures.
    amvh_body[0x2C..0x30].copy_from_slice(&1u32.to_le_bytes());
    // Reserved zero at +0x30 (kept implicitly).
    // Packed duration at +0x34.
    amvh_body[0x34..0x38].copy_from_slice(&duration_packed.to_le_bytes());
    buf.extend_from_slice(&amvh_body);
    // ── §3a Video strl: LIST 0 strl strh <0x38 all-zero> strf <0x24 all-zero> ──
    buf.extend_from_slice(b"LIST");
    buf.extend_from_slice(&[0u8; 4]); // strl LIST size — zeroed.
    buf.extend_from_slice(b"strl");
    buf.extend_from_slice(b"strh");
    buf.extend_from_slice(&STRH_VIDEO_BODY_LEN.to_le_bytes());
    buf.extend_from_slice(&vec![0u8; STRH_VIDEO_BODY_LEN as usize]);
    buf.extend_from_slice(b"strf");
    buf.extend_from_slice(&STRF_VIDEO_BODY_LEN.to_le_bytes());
    buf.extend_from_slice(&vec![0u8; STRF_VIDEO_BODY_LEN as usize]);
    // ── §3b Audio strl: LIST 0 strl strh <0x30 all-zero> strf <0x14 WAVEFORMATEX> ──
    buf.extend_from_slice(b"LIST");
    buf.extend_from_slice(&[0u8; 4]);
    buf.extend_from_slice(b"strl");
    buf.extend_from_slice(b"strh");
    buf.extend_from_slice(&STRH_AUDIO_BODY_LEN.to_le_bytes());
    buf.extend_from_slice(&vec![0u8; STRH_AUDIO_BODY_LEN as usize]);
    buf.extend_from_slice(b"strf");
    buf.extend_from_slice(&STRF_AUDIO_BODY_LEN.to_le_bytes());
    let mut strf_audio = vec![0u8; STRF_AUDIO_BODY_LEN as usize];
    // WAVEFORMATEX layout from §3b:
    //   +0x00 u16 wFormatTag        = 1 (PCM declared; observed convention)
    //   +0x02 u16 nChannels         = `channels` (mono in fixtures)
    //   +0x04 u32 nSamplesPerSec    = `samples_per_sec`
    //   +0x08 u32 nAvgBytesPerSec   = samples_per_sec * 2 (decoded-PCM rate)
    //   +0x0C u16 nBlockAlign       = 2
    //   +0x0E u16 wBitsPerSample    = 16
    //   +0x10 u16 cbSize            = 0
    strf_audio[0x00..0x02].copy_from_slice(&1u16.to_le_bytes());
    strf_audio[0x02..0x04].copy_from_slice(&channels.to_le_bytes());
    strf_audio[0x04..0x08].copy_from_slice(&samples_per_sec.to_le_bytes());
    strf_audio[0x08..0x0C].copy_from_slice(&samples_per_sec.saturating_mul(2).to_le_bytes());
    strf_audio[0x0C..0x0E].copy_from_slice(&2u16.to_le_bytes());
    strf_audio[0x0E..0x10].copy_from_slice(&16u16.to_le_bytes());
    strf_audio[0x10..0x12].copy_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(&strf_audio);
    // ── §4 movi LIST opener ───────────────────────────────────────
    buf.extend_from_slice(b"LIST");
    buf.extend_from_slice(&[0u8; 4]); // movi LIST size — zeroed.
    buf.extend_from_slice(b"movi");
    debug_assert_eq!(buf.len(), 0x13C);
    buf
}

/// Pack `(seconds, minutes, hours)` into the little-endian u32 written
/// at `amvh +0x34` per §2: bytes laid out `[seconds, minutes, hours, 0]`.
fn pack_duration(seconds: u8, minutes: u8, hours: u8) -> u32 {
    u32::from_le_bytes([seconds, minutes, hours, 0])
}

/// Factory matching `oxideav_core::OpenMuxerFn`.
pub(crate) fn open_muxer(
    output: Box<dyn WriteSeek>,
    streams: &[StreamInfo],
) -> Result<Box<dyn Muxer>> {
    Ok(Box::new(AmvMuxer::open(output, streams)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::AmvDuration;
    use crate::{AUDIO_CODEC_ID, VIDEO_CODEC_ID};
    use oxideav_core::{
        CodecId, CodecParameters, Demuxer as _, MediaType, PixelFormat, Rational, StreamInfo,
        TimeBase,
    };
    use std::io::Cursor;

    /// Build a `[StreamInfo; 2]` matching the `comedian.amv` parameters
    /// so muxer tests + roundtrip tests share one source of truth.
    fn comedian_streams() -> Vec<StreamInfo> {
        let mut video_params = CodecParameters::video(CodecId::new(VIDEO_CODEC_ID));
        video_params.media_type = MediaType::Video;
        video_params.width = Some(128);
        video_params.height = Some(96);
        video_params.pixel_format = Some(PixelFormat::Yuv420P);
        video_params.frame_rate = Some(Rational::new(12, 1));
        let video = StreamInfo {
            index: 0,
            time_base: TimeBase::new(1, 12),
            duration: None,
            start_time: Some(0),
            params: video_params,
        };
        let mut audio_params = CodecParameters::audio(CodecId::new(AUDIO_CODEC_ID));
        audio_params.media_type = MediaType::Audio;
        audio_params.sample_rate = Some(22_050);
        audio_params.channels = Some(1);
        let audio = StreamInfo {
            index: 1,
            time_base: TimeBase::new(1, 22_050),
            duration: None,
            start_time: Some(0),
            params: audio_params,
        };
        vec![video, audio]
    }

    #[test]
    fn pack_duration_matches_comedian() {
        // §2 worked example: 33 s + 1 min + 0 h → 0x00 00 01 21 LE.
        assert_eq!(pack_duration(0x21, 0x01, 0x00), 0x0000_0121);
    }

    #[test]
    fn pack_duration_matches_noel() {
        // §2: 2 s + 3 min + 0 h → 0x00 00 03 02 LE.
        assert_eq!(pack_duration(2, 3, 0), 0x0000_0302);
    }

    #[test]
    fn build_prelude_bytes_match_demuxer_constants() {
        // Cross-check that the muxer's hardcoded body lengths still
        // match the demuxer's expectations (parse.rs private consts).
        // If they ever drift, `AmvPrelude::parse` would reject the
        // output and this test would fail via the roundtrip test —
        // surface the mismatch more directly here too.
        let buf = build_prelude_bytes(128, 96, 12, 0x0000_0121, 22_050, 1);
        assert_eq!(buf.len(), 0x13C);
        assert_eq!(&buf[0x00..0x04], b"RIFF");
        assert_eq!(&buf[0x08..0x0C], b"AMV ");
        assert_eq!(&buf[0x0C..0x10], b"LIST");
        assert_eq!(&buf[0x14..0x18], b"hdrl");
        assert_eq!(&buf[0x18..0x1C], b"amvh");
        // amvh +0x00 = micros_per_frame.
        let micros = u32::from_le_bytes(buf[0x20..0x24].try_into().unwrap());
        assert_eq!(micros, 1_000_000 / 12);
        // amvh +0x20 = width (file 0x40).
        let width = u32::from_le_bytes(buf[0x40..0x44].try_into().unwrap());
        assert_eq!(width, 128);
        // amvh +0x34 = packed duration (file 0x54).
        let dur = u32::from_le_bytes(buf[0x54..0x58].try_into().unwrap());
        assert_eq!(dur, 0x0000_0121);
        // movi opener tail.
        assert_eq!(&buf[0x130..0x134], b"LIST");
        assert_eq!(&buf[0x138..0x13C], b"movi");
    }

    #[test]
    fn open_rejects_wrong_stream_count() {
        let streams = vec![comedian_streams().pop().unwrap()];
        let writer: Box<dyn WriteSeek> = Box::new(Cursor::new(Vec::<u8>::new()));
        assert!(AmvMuxer::open(writer, &streams).is_err());
    }

    #[test]
    fn open_rejects_swapped_stream_order() {
        let mut streams = comedian_streams();
        streams.swap(0, 1);
        // Indices on the StreamInfo still report 0 and 1; the muxer
        // requires stream-array[0].index == 0 specifically.
        let writer: Box<dyn WriteSeek> = Box::new(Cursor::new(Vec::<u8>::new()));
        assert!(AmvMuxer::open(writer, &streams).is_err());
    }

    #[test]
    fn open_rejects_missing_video_dimensions() {
        let mut streams = comedian_streams();
        streams[0].params.width = None;
        let writer: Box<dyn WriteSeek> = Box::new(Cursor::new(Vec::<u8>::new()));
        assert!(AmvMuxer::open(writer, &streams).is_err());
    }

    #[test]
    fn open_rejects_missing_audio_sample_rate() {
        let mut streams = comedian_streams();
        streams[1].params.sample_rate = None;
        let writer: Box<dyn WriteSeek> = Box::new(Cursor::new(Vec::<u8>::new()));
        assert!(AmvMuxer::open(writer, &streams).is_err());
    }

    #[test]
    fn write_packet_before_header_is_error() {
        let streams = comedian_streams();
        let writer: Box<dyn WriteSeek> = Box::new(Cursor::new(Vec::<u8>::new()));
        let mut mux = AmvMuxer::open(writer, &streams).unwrap();
        let pkt = Packet::new(0, TimeBase::new(1, 12), vec![0xFF, 0xD8, 0xFF, 0xD9]);
        assert!(mux.write_packet(&pkt).is_err());
    }

    /// Build a synthetic round-trip: assemble 3 (video, audio) pairs
    /// into a buffer using exactly the same byte sequence the muxer
    /// produces, then demux and check that the prelude + chunks +
    /// trailer all parse back to the inputs.
    ///
    /// The helper [`build_via_no_box_muxer`] replicates the muxer's
    /// writes onto a `Cursor<Vec<u8>>` we keep ownership of — the
    /// `dyn WriteSeek` trait erases the underlying writer's concrete
    /// type so a real `AmvMuxer` instance can't hand the bytes back
    /// for inspection. A separate test
    /// ([`writer_byte_sequence_matches_no_box_helper`]) pins the two
    /// code paths to identical byte output.
    #[test]
    fn roundtrip_mux_then_demux_recovers_packets() {
        use crate::demuxer::AmvDemuxer;

        let streams = comedian_streams();
        // 3 pairs of (00dc 4-byte payload, 01wb 12-byte payload).
        let video_payloads: Vec<Vec<u8>> = (0..3).map(|i: u32| i.to_le_bytes().to_vec()).collect();
        let mut audio_payloads: Vec<Vec<u8>> = Vec::new();
        for _ in 0..3 {
            // 8-byte preamble (state=0, decoded_samples=1837) + 4 nibble bytes.
            let mut a = Vec::with_capacity(12);
            a.extend_from_slice(&0u32.to_le_bytes());
            a.extend_from_slice(&1837u32.to_le_bytes());
            a.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
            audio_payloads.push(a);
        }
        let direct = build_via_no_box_muxer(&streams, &video_payloads, &audio_payloads);
        let mut d = AmvDemuxer::open(Cursor::new(direct)).unwrap();
        assert_eq!(d.streams().len(), 2);
        assert_eq!(d.header().width, 128);
        assert_eq!(d.header().height, 96);
        assert_eq!(d.header().fps, 12);
        // 3 frames @ 12 fps = 0 seconds (integer floor). Duration
        // patching still runs and writes 0x00.
        assert_eq!(d.audio_format().samples_per_sec, 22_050);

        let mut n_video = 0u32;
        let mut n_audio = 0u32;
        let mut recovered_video: Vec<Vec<u8>> = Vec::new();
        let mut recovered_audio: Vec<Vec<u8>> = Vec::new();
        loop {
            match d.next_packet() {
                Ok(p) if p.stream_index == 0 => {
                    recovered_video.push(p.data.clone());
                    n_video += 1;
                }
                Ok(p) if p.stream_index == 1 => {
                    recovered_audio.push(p.data.clone());
                    n_audio += 1;
                }
                Ok(_) => panic!("unexpected stream index"),
                Err(Error::Eof) => break,
                Err(e) => panic!("walk error: {e:?}"),
            }
        }
        assert_eq!(n_video, 3);
        assert_eq!(n_audio, 3);
        assert_eq!(recovered_video, video_payloads);
        assert_eq!(recovered_audio, audio_payloads);
    }

    /// Mux the same prelude + chunks + trailer into a fresh
    /// `Cursor<Vec<u8>>` we keep ownership of, so we can inspect the
    /// bytes. Duplicates the muxer's behaviour with a non-boxed writer.
    fn build_via_no_box_muxer(
        streams: &[StreamInfo],
        videos: &[Vec<u8>],
        audios: &[Vec<u8>],
    ) -> Vec<u8> {
        let mut cur = Cursor::new(Vec::<u8>::new());
        // Manually replicate the AmvMuxer's writes onto a non-boxed
        // Cursor so we can recover the underlying Vec at the end.
        let prelude = build_prelude_bytes(
            streams[0].params.width.unwrap(),
            streams[0].params.height.unwrap(),
            streams[0].params.frame_rate.unwrap().num as u32,
            0,
            streams[1].params.sample_rate.unwrap(),
            streams[1].params.channels.unwrap(),
        );
        cur.write_all(&prelude).unwrap();
        let mut frames = 0u64;
        for i in 0..videos.len().max(audios.len()) {
            if i < videos.len() {
                cur.write_all(&VIDEO_CHUNK_TAG).unwrap();
                cur.write_all(&(videos[i].len() as u32).to_le_bytes())
                    .unwrap();
                cur.write_all(&videos[i]).unwrap();
                frames += 1;
            }
            if i < audios.len() {
                cur.write_all(&AUDIO_CHUNK_TAG).unwrap();
                cur.write_all(&(audios[i].len() as u32).to_le_bytes())
                    .unwrap();
                cur.write_all(&audios[i]).unwrap();
            }
        }
        cur.write_all(&AMV_END_TRAILER).unwrap();
        let fps = streams[0].params.frame_rate.unwrap().num as u64;
        if fps > 0 && frames > 0 {
            let total_seconds = frames / fps;
            let packed = pack_duration(
                (total_seconds % 60) as u8,
                ((total_seconds % 3600) / 60) as u8,
                (total_seconds / 3600).min(255) as u8,
            );
            cur.seek(SeekFrom::Start(AMVH_DURATION_FILE_OFFSET))
                .unwrap();
            cur.write_all(&packed.to_le_bytes()).unwrap();
        }
        cur.into_inner()
    }

    #[test]
    fn write_trailer_patches_duration_to_comedian_value() {
        // 1116 frames @ 12 fps = 93 s = 1:33 → 0x0000_0121.
        let streams = comedian_streams();
        let videos: Vec<Vec<u8>> = (0..1116u32).map(|_| vec![0xFF, 0xD8, 0xFF, 0xD9]).collect();
        let audios: Vec<Vec<u8>> = (0..1116u32)
            .map(|_| {
                let mut a = Vec::with_capacity(8);
                a.extend_from_slice(&0u32.to_le_bytes());
                a.extend_from_slice(&1837u32.to_le_bytes());
                a
            })
            .collect();
        let buf = build_via_no_box_muxer(&streams, &videos, &audios);
        // Inspect the amvh +0x34 dword.
        let packed = u32::from_le_bytes(buf[0x54..0x58].try_into().unwrap());
        assert_eq!(packed, 0x0000_0121);
        // Round-trip parses cleanly with the right duration.
        let d = crate::demuxer::AmvDemuxer::open(Cursor::new(buf)).unwrap();
        let dur = d.header().duration();
        assert_eq!(
            dur,
            AmvDuration {
                seconds: 0x21,
                minutes: 1,
                hours: 0
            }
        );
        assert_eq!(d.duration_micros(), Some(93_000_000));
    }

    /// A writer that owns the underlying `Vec<u8>` behind an
    /// `Arc<Mutex<Vec<u8>>>` so the test can recover the bytes after
    /// the boxed muxer has been dropped. Pure test scaffolding —
    /// `AmvMuxer` itself does not need this; production callers can
    /// own a `File` outright and re-open it for reading.
    #[derive(Clone)]
    struct SharedCursor(std::sync::Arc<std::sync::Mutex<Cursor<Vec<u8>>>>);

    impl Write for SharedCursor {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().write(buf)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.0.lock().unwrap().flush()
        }
    }

    impl Seek for SharedCursor {
        fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
            self.0.lock().unwrap().seek(pos)
        }
    }

    /// Pin the real `AmvMuxer` to the byte sequence produced by the
    /// in-test [`build_via_no_box_muxer`] helper, so the round-trip
    /// test above is a faithful proxy for what a real boxed muxer
    /// emits over the trait.
    #[test]
    fn writer_byte_sequence_matches_no_box_helper() {
        let streams = comedian_streams();
        let video_payloads: Vec<Vec<u8>> = (0..3).map(|i: u32| i.to_le_bytes().to_vec()).collect();
        let audio_payloads: Vec<Vec<u8>> = (0..3)
            .map(|_| {
                let mut a = Vec::with_capacity(12);
                a.extend_from_slice(&0u32.to_le_bytes());
                a.extend_from_slice(&1837u32.to_le_bytes());
                a.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
                a
            })
            .collect();

        let shared = SharedCursor(std::sync::Arc::new(std::sync::Mutex::new(Cursor::new(
            Vec::<u8>::new(),
        ))));
        let writer: Box<dyn WriteSeek> = Box::new(shared.clone());
        let mut mux = AmvMuxer::open(writer, &streams).unwrap();
        mux.write_header().unwrap();
        for i in 0..3 {
            mux.write_packet(&Packet::new(
                0,
                TimeBase::new(1, 12),
                video_payloads[i].clone(),
            ))
            .unwrap();
            mux.write_packet(&Packet::new(
                1,
                TimeBase::new(1, 22_050),
                audio_payloads[i].clone(),
            ))
            .unwrap();
        }
        mux.write_trailer().unwrap();
        drop(mux);

        let real = shared.0.lock().unwrap().get_ref().clone();
        let helper = build_via_no_box_muxer(&streams, &video_payloads, &audio_payloads);
        assert_eq!(
            real, helper,
            "AmvMuxer output diverges from build_via_no_box_muxer"
        );
    }

    #[test]
    fn build_via_no_box_muxer_handles_noel_values() {
        // §2 cross-check: noel-son-lumiere — 96×64 @ 16 fps, 2928 frames
        // → 183 s = 3 min 3 s. Sample data has 2928÷16=183s = 3:03 but
        // the recorded packed value in the trace is `02 03 00 00`
        // (3:02). The discrepancy is in the source's written duration;
        // we exercise the *encoder* logic here (which derives from
        // frame count) so expect 3:03 = `03 03 00 00`.
        let mut streams = comedian_streams();
        streams[0].params.width = Some(96);
        streams[0].params.height = Some(64);
        streams[0].params.frame_rate = Some(Rational::new(16, 1));
        let videos: Vec<Vec<u8>> = (0..2928u32).map(|_| vec![0; 8]).collect();
        let audios: Vec<Vec<u8>> = (0..2928u32)
            .map(|_| {
                let mut a = Vec::new();
                a.extend_from_slice(&0u32.to_le_bytes());
                a.extend_from_slice(&1378u32.to_le_bytes());
                a
            })
            .collect();
        let buf = build_via_no_box_muxer(&streams, &videos, &audios);
        let d = crate::demuxer::AmvDemuxer::open(Cursor::new(buf)).unwrap();
        assert_eq!(d.header().width, 96);
        assert_eq!(d.header().height, 64);
        assert_eq!(d.header().fps, 16);
        let dur = d.header().duration();
        // 2928 / 16 = 183 → 3 min 3 s.
        assert_eq!(dur.seconds, 3);
        assert_eq!(dur.minutes, 3);
        assert_eq!(dur.hours, 0);
    }
}
