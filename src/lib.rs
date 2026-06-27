//! # oxideav-amv
//!
//! Pure-Rust demuxer for the **AMV** ("Actions Media Video") container —
//! the non-standard AVI variant used by inexpensive Chinese "MP3/MP4"
//! portable media players (S1 / Actions / ALi-chip devices).
//!
//! AMV pairs a custom intra-only Motion-JPEG-like video stream (fixed,
//! hardcoded quant/Huffman tables) with an IMA-ADPCM-style mono audio
//! stream, wrapped in a `RIFF` form whose type is `AMV ` (trailing
//! space). It reuses the AVI 1.0 RIFF vocabulary (`LIST` / `hdrl` /
//! `strl` / `strh` / `strf` / `movi`, the `<nn>dc` / `<nn>wb` chunk-tag
//! convention, `WAVEFORMATEX`) while discarding most of its semantics:
//! all RIFF / LIST sizes are zeroed, leaf chunks are **not** padded to
//! an even byte boundary, stream-header bodies are mostly blank, there
//! is no `idx1` index, and the stream is bounded by an `AMV_END_` ASCII
//! trailer instead of a RIFF chunk length.
//!
//! ## What this crate decodes
//!
//! It identifies AMV files, parses the `amvh` main header (resolution,
//! fps, packed duration) and audio `WAVEFORMATEX`, walks the `movi`
//! payload, and emits one [`Packet`](oxideav_core::Packet) per `00dc`
//! (video) / `01wb` (audio) leaf chunk. Because AMV's video and audio
//! profiles are **intrinsic to the device** (the table-stripped JPEG and
//! the IMA-ADPCM-with-an-8-byte-preamble block both have no other home —
//! every AMV file shares the one fixed profile), the crate also **owns
//! the codecs**: it registers real `oxideav-core` `Decoder` / `Encoder`
//! factories under the `amv_video` (video) and `adpcm_amv` (audio) ids,
//! and the demuxer declares those ids on its streams so a registered
//! `RuntimeContext` resolves a working decoder straight from the demuxer
//! output. [`AmvVideoFrame`] is the typed geometry-binding surface those
//! decoders consume: it binds the `amvh` geometry (the only place the
//! resolution exists — the `00dc` bitstream carries no frame header) to a
//! validated `00dc` payload and exposes the entropy-coded window.
//!
//! The container also still declares the [`VIDEO_CODEC_ID`] (`mjpeg`)
//! convenience id for the *reconstruct-then-mjpeg* route — splice the
//! §4a tables back with [`reconstruct_jpeg`] and hand the conforming
//! JFIF to a generic decoder — but a pipeline opening an `.amv` resolves
//! the [`VIDEO_DIRECT_CODEC_ID`] (`amv_video`) stream id to the in-crate
//! direct decoder, which a generic `mjpeg` decoder cannot replace.
//!
//! ## What this crate writes
//!
//! [`AmvMuxer`] is the inverse of [`AmvDemuxer`]: given the same
//! `[video, audio]` `StreamInfo` pair, it emits a byte-faithful AMV
//! file with the §1 zeroed RIFF / LIST sizes, the §2 packed-byte
//! `amvh` duration (patched from observed frame count in
//! `write_trailer`), the §3 all-zero stream-header bodies plus the
//! 20-byte audio `WAVEFORMATEX`, the §4 no-byte-padding chunk walk,
//! and the §4c `AMV_END_` ASCII trailer. Mux → demux round-trips
//! exactly.
//!
//! The device's intrinsic codecs are also **encoded** here, the
//! byte-inverse of the decode paths: [`encode_frame_rgb`] turns an
//! upright RGB raster into a bare `00dc` baseline-JPEG payload
//! (§4a device tables, 4:2:0, table-stripped) and [`encode_audio_payload`]
//! turns 16-bit mono PCM into an `01wb` IMA-ADPCM block (§4b). Feeding
//! their output to [`AmvMuxer`] produces a complete AMV file, so a real
//! `.amv` round-trips decode → encode → mux → demux → decode.
//!
//! ## Provenance
//!
//! Every byte-format fact in this crate is derived from
//! `docs/container/amv/amv-container-trace.md` in the workspace,
//! which was reverse-engineered from observed bytes of redistributable
//! `.amv` samples. No external multimedia-library source code or any
//! third-party AMV demuxer was consulted. Field names such as
//! `amvh`, `strh`, `strf`, `movi`, `00dc`, `01wb` are the literal
//! four-byte ASCII identifiers observed in the file; their structural
//! roles follow Microsoft's published RIFF/AVI references.
//!
//! ## Example
//!
//! ```no_run
//! use std::fs::File;
//! use std::io::BufReader;
//! use oxideav_amv::AmvDemuxer;
//!
//! let f = File::open("sample.amv").unwrap();
//! let mut demuxer = AmvDemuxer::open(BufReader::new(f)).unwrap();
//! assert_eq!(demuxer.header().width, 128);
//! assert_eq!(demuxer.header().height, 96);
//! ```
//!
//! When wired into an `oxideav_core::RuntimeContext` via
//! [`register`], `amv` becomes available alongside the other registered
//! containers and can be opened through the registry-driven probe path.

#![warn(missing_debug_implementations)]
#![warn(missing_docs)]

mod adpcm;
mod adpcm_encode;
mod codec_audio;
mod codec_video;
mod demuxer;
mod jpeg_decode;
mod jpeg_encode;
mod jpeg_reconstruct;
mod muxer;
mod parse;
mod video;

pub use adpcm::{decode_audio_block, decode_audio_payload};
pub use adpcm_encode::{encode_audio_nibbles, encode_audio_payload};
pub use codec_audio::{
    make_decoder as make_audio_decoder, make_encoder as make_audio_encoder, AmvAudioDecoder,
    AmvAudioEncoder, AMV_AUDIO_CHANNELS, AMV_AUDIO_SAMPLE_RATE,
};
pub use codec_video::{
    make_decoder as make_video_decoder, make_encoder as make_video_encoder, AmvVideoDecoder,
    AmvVideoEncoder,
};
pub use demuxer::{AmvDemuxer, AmvDemuxerError, ChunkIndexEntry};
pub use jpeg_decode::{
    decode_frame, decode_frame_from_payload, decode_frame_yuv420p,
    decode_frame_yuv420p_from_payload, DecodedFrame, DecodedYuv420p,
};
pub use jpeg_encode::{encode_frame, encode_frame_rgb, encode_frame_yuv420p};
pub use jpeg_reconstruct::{reconstruct_jpeg, reconstruct_jpeg_from_payload};
pub use muxer::AmvMuxer;
pub use parse::{
    validate_movi_interleave, validate_video_payload_no_internal_markers,
    validate_video_payload_shape, AmvAudioPreamble, AmvDuration, AmvHeader, AmvWaveFormat,
    ChunkHeader, ChunkKind, MoviPayload, MoviPayloadIter, AMVH_BODY_LEN, AMV_AUDIO_PREAMBLE_LEN,
    AMV_END_TRAILER, AMV_FORM_TYPE, AUDIO_CHUNK_TAG, IMA_STEP_INDEX_MAX, JPEG_EOI, JPEG_SOI,
    VIDEO_CHUNK_TAG,
};
pub use video::{flip_rows_vertical, AmvVideoFrame};

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecRegistry, CodecResolver, ContainerRegistry,
    Demuxer, Error, ProbeData, ProbeScore, ReadSeek, Result, RuntimeContext,
};

/// Codec id string declared for the AMV video stream. Maps to the
/// existing `mjpeg` codec — AMV's video is an intra-only JPEG variant
/// with hardcoded quant / Huffman tables stripped from the bitstream.
///
/// The actual decoder for AMV's particular table-stripped JPEG variant
/// is a separate concern (the AMV player splices its tables back in
/// before handing the bitstream to a standard JPEG decoder); declaring
/// `mjpeg` here lets downstream consumers route the packet to MJPEG
/// machinery once an AMV-aware preprocessor is wired up.
pub const VIDEO_CODEC_ID: &str = "mjpeg";

/// Codec id string for the AMV **direct** video codec — the
/// table-stripped baseline-JPEG decoder/encoder this crate registers
/// for the bare `00dc` bitstream (trace §4a).
///
/// Distinct from [`VIDEO_CODEC_ID`] (`mjpeg`): the generic `mjpeg`
/// decoder cannot consume a `00dc` payload because its quant / Huffman
/// tables and scan geometry are stripped from the wire and hardcoded in
/// the device — they have to be spliced back in (the
/// [`reconstruct_jpeg`] path) before a standard JPEG decoder accepts the
/// stream. `amv_video` decodes the bare bitstream directly against the
/// §4a device profile, so it is registered under its own id rather than
/// colliding with the `mjpeg` id `oxideav-mjpeg` owns. A consumer that
/// wants the reconstruct-then-mjpeg route still uses `mjpeg`; a consumer
/// that wants the one-step native decode uses `amv_video`.
pub const VIDEO_DIRECT_CODEC_ID: &str = "amv_video";

/// Codec id string declared for the AMV audio stream. Reserved for a
/// future ADPCM variant — AMV's audio is a 4-bit nibble codec with an
/// 8-byte per-chunk preamble (decoded sample count + initial state) and
/// is **not** the WAVEFORMATEX `wFormatTag = 0x0001` PCM the header
/// claims. The container declares this id so a downstream
/// `adpcm_amv` codec can register against it without an extra
/// reconciliation pass.
pub const AUDIO_CODEC_ID: &str = "adpcm_amv";

/// Container format name registered with [`ContainerRegistry`].
pub const CONTAINER_NAME: &str = "amv";

/// Magic bytes used by the probe: the first 12 bytes of every AMV file
/// are `RIFF <size:4> 'AMV '` (size is always zero in the wild but the
/// probe tolerates any value — only the FORM type is signal).
const PROBE_MAGIC_RIFF: &[u8; 4] = b"RIFF";
const PROBE_MAGIC_AMV: &[u8; 4] = b"AMV ";

/// Probe function: score `100` when the first 12 bytes match
/// `RIFF .... 'AMV '`, `0` otherwise. The trailing space inside `AMV `
/// is the FOURCC's fourth byte and is the only thing distinguishing AMV
/// from a conforming AVI file at this offset.
pub fn probe(p: &ProbeData) -> ProbeScore {
    if p.buf.len() < 12 {
        return 0;
    }
    if &p.buf[0..4] != PROBE_MAGIC_RIFF {
        return 0;
    }
    if &p.buf[8..12] != PROBE_MAGIC_AMV {
        return 0;
    }
    100
}

/// Open an AMV demuxer from a seekable reader. Used as the factory by
/// [`register_containers`].
fn open_demuxer(input: Box<dyn ReadSeek>, _codecs: &dyn CodecResolver) -> Result<Box<dyn Demuxer>> {
    let demuxer = AmvDemuxer::open(input).map_err(Error::from)?;
    Ok(Box::new(demuxer))
}

/// Register the AMV container into a [`ContainerRegistry`]. Installs
/// both the demuxer (read side) and the muxer (write side); the writer
/// is byte-faithful to `docs/container/amv/amv-container-trace.md`
/// §1..§4 so mux→demux round-trips cleanly.
pub fn register_containers(reg: &mut ContainerRegistry) {
    reg.register_demuxer(CONTAINER_NAME, open_demuxer);
    reg.register_muxer(CONTAINER_NAME, muxer::open_muxer);
    reg.register_extension("amv", CONTAINER_NAME);
    reg.register_probe(CONTAINER_NAME, probe);
}

/// Register the AMV stream codecs into a [`CodecRegistry`].
///
/// Both intrinsic device codecs install real decoder + encoder
/// factories:
///
/// * **`adpcm_amv`** (audio) — AMV's IMA-ADPCM-with-an-8-byte-preamble
///   audio (trace §3b / §4b), mono / 22 050 Hz / S16.
/// * **`amv_video`** (video) — the table-stripped §4a baseline JPEG,
///   decoded / encoded directly to / from YUV420P. This is *distinct*
///   from the `mjpeg` id ([`VIDEO_CODEC_ID`]) the demuxer also declares:
///   a generic `mjpeg` decoder cannot consume a bare `00dc` payload (its
///   tables are stripped), so `amv_video` is registered under its own id
///   and the `mjpeg` id is left to `oxideav-mjpeg` for the
///   reconstruct-then-decode route. See [`VIDEO_DIRECT_CODEC_ID`].
///
/// Both are intrinsic to the AMV device profile and have no other home,
/// so they are owned here (crate-purpose discipline: codecs with
/// dedicated native containers). The lower-level free functions
/// (`decode_frame` / `encode_frame_rgb`, `decode_audio_payload` /
/// `encode_audio_payload`) remain available for callers driving the
/// bitstream by hand.
pub fn register_codecs(reg: &mut CodecRegistry) {
    // Audio: adpcm_amv (trace §3b/§4b).
    let audio_caps = CodecCapabilities::audio("adpcm_amv")
        .with_decode()
        .with_encode()
        .with_lossy(true)
        .with_max_channels(AMV_AUDIO_CHANNELS)
        .with_max_sample_rate(AMV_AUDIO_SAMPLE_RATE);
    reg.register(
        CodecInfo::new(CodecId::new(AUDIO_CODEC_ID))
            .capabilities(audio_caps)
            .decoder(codec_audio::make_decoder)
            .encoder(codec_audio::make_encoder),
    );

    // Video: amv_video — the table-stripped §4a baseline JPEG, decoded
    // / encoded directly (not via the generic `mjpeg` reconstruct path).
    let video_caps = CodecCapabilities::video("amv_video")
        .with_decode()
        .with_encode()
        .with_intra_only(true)
        .with_lossy(true)
        .with_pixel_format(oxideav_core::PixelFormat::Yuv420P);
    reg.register(
        CodecInfo::new(CodecId::new(VIDEO_DIRECT_CODEC_ID))
            .capabilities(video_caps)
            .decoder(codec_video::make_decoder)
            .encoder(codec_video::make_encoder),
    );
}

/// Crate-local error mirror for the framework's `Error`.
impl From<AmvDemuxerError> for Error {
    fn from(e: AmvDemuxerError) -> Self {
        match e {
            AmvDemuxerError::Io(s) => Error::other(s),
            AmvDemuxerError::InvalidData(s) => Error::invalid(s),
            AmvDemuxerError::Eof => Error::Eof,
        }
    }
}

/// `oxideav-core` entry point: install AMV's container and (no-op)
/// codec registrations into a [`RuntimeContext`].
pub fn register(ctx: &mut RuntimeContext) {
    register_containers(&mut ctx.containers);
    register_codecs(&mut ctx.codecs);
}

oxideav_core::register!("amv", register);

#[cfg(test)]
mod register_tests {
    use super::*;
    use oxideav_core::RuntimeContext;

    #[test]
    fn register_installs_amv_demuxer() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        assert!(
            ctx.containers.demuxer_names().any(|n| n == CONTAINER_NAME),
            "amv demuxer should be installed"
        );
    }

    #[test]
    fn register_installs_amv_muxer() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        assert!(
            ctx.containers.muxer_names().any(|n| n == CONTAINER_NAME),
            "amv muxer should be installed"
        );
    }

    #[test]
    fn register_installs_adpcm_amv_codec() {
        use oxideav_core::{CodecId, CodecParameters, SampleFormat};
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let id = CodecId::new(AUDIO_CODEC_ID);
        assert!(
            ctx.codecs.has_decoder(&id),
            "adpcm_amv decoder should be installed"
        );
        assert!(
            ctx.codecs.has_encoder(&id),
            "adpcm_amv encoder should be installed"
        );
        // The registry factories build through the same path as the
        // direct `make_audio_*` entry points.
        let mut params = CodecParameters::audio(id);
        params.channels = Some(1);
        params.sample_format = Some(SampleFormat::S16);
        params.sample_rate = Some(AMV_AUDIO_SAMPLE_RATE);
        assert!(ctx.codecs.first_decoder(&params).is_ok());
        assert!(ctx.codecs.first_encoder(&params).is_ok());
    }

    #[test]
    fn register_installs_amv_video_codec() {
        use oxideav_core::{CodecId, CodecParameters, PixelFormat};
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let id = CodecId::new(VIDEO_DIRECT_CODEC_ID);
        assert!(
            ctx.codecs.has_decoder(&id),
            "amv_video decoder should be installed"
        );
        assert!(
            ctx.codecs.has_encoder(&id),
            "amv_video encoder should be installed"
        );
        let mut params = CodecParameters::video(id);
        params.width = Some(128);
        params.height = Some(96);
        params.pixel_format = Some(PixelFormat::Yuv420P);
        assert!(ctx.codecs.first_decoder(&params).is_ok());
        assert!(ctx.codecs.first_encoder(&params).is_ok());
    }

    #[test]
    fn probe_matches_riff_amv_signature() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&[0; 4]);
        buf.extend_from_slice(b"AMV ");
        assert_eq!(
            probe(&ProbeData {
                buf: &buf,
                ext: None
            }),
            100
        );
    }

    #[test]
    fn probe_rejects_avi_form_type() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&[0; 4]);
        buf.extend_from_slice(b"AVI ");
        assert_eq!(
            probe(&ProbeData {
                buf: &buf,
                ext: None
            }),
            0
        );
    }

    #[test]
    fn probe_rejects_short_input() {
        assert_eq!(
            probe(&ProbeData {
                buf: b"RIFF",
                ext: None
            }),
            0
        );
    }
}
