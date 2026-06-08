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
//! This is a **container-only** crate: it identifies AMV files, parses
//! the `amvh` main header (resolution, fps, packed duration) and audio
//! `WAVEFORMATEX`, walks the `movi` payload, and emits one
//! [`Packet`](oxideav_core::Packet) per `00dc` (video) / `01wb` (audio)
//! leaf chunk. The video stream is declared as `mjpeg` and the audio
//! stream as `adpcm_amv`; actual frame / sample decoding lives in
//! sibling codec crates and is invoked downstream of the demuxer.
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

mod demuxer;
mod muxer;
mod parse;

pub use demuxer::{AmvDemuxer, AmvDemuxerError, ChunkIndexEntry};
pub use muxer::AmvMuxer;
pub use parse::{
    validate_movi_interleave, validate_video_payload_no_internal_markers,
    validate_video_payload_shape, AmvAudioPreamble, AmvDuration, AmvHeader, AmvWaveFormat,
    ChunkHeader, ChunkKind, MoviPayload, MoviPayloadIter, AMVH_BODY_LEN, AMV_AUDIO_PREAMBLE_LEN,
    AMV_END_TRAILER, AMV_FORM_TYPE, AUDIO_CHUNK_TAG, JPEG_EOI, JPEG_SOI, VIDEO_CHUNK_TAG,
};

use oxideav_core::{
    CodecRegistry, CodecResolver, ContainerRegistry, Demuxer, Error, ProbeData, ProbeScore,
    ReadSeek, Result, RuntimeContext,
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

/// Register the AMV stream-codec placeholder ids into a
/// [`CodecRegistry`].
///
/// Neither id installs a factory — AMV is a container, not a codec.
/// The registration exists so that downstream codec crates can later
/// attach decoders / encoders against `mjpeg` (already wired by
/// `oxideav-mjpeg`) and `adpcm_amv` (reserved for a future
/// AMV-IMA-ADPCM variant) without an extra mapping step.
pub fn register_codecs(reg: &mut CodecRegistry) {
    // The video tag is the `mjpeg` codec id; do not re-register the
    // codec itself here (the dedicated `oxideav-mjpeg` crate owns
    // that). Only the audio side gets a placeholder entry so a future
    // ADPCM-AMV crate can register against it.
    let _ = reg;
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
