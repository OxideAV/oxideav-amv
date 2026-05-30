# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Clean-room AMV container demuxer rebuilt against
  `docs/container/amv/amv-container-trace.md`, the workspace-staged
  byte-level reverse-engineering trace (no external library source
  consulted, no web access).
- Standalone byte parsers — `AmvHeader` (`amvh` body: `dwMicroSecPerFrame`,
  width, height, fps, byte-packed duration), `AmvDuration` (the
  `[seconds, minutes, hours, 0]` packed-byte unpacker), `AmvWaveFormat` (the
  audio `strf` `WAVEFORMATEX`), `ChunkHeader` + `ChunkKind` (leaf-chunk
  framing with the AMV no-padding rule).
- `AmvDemuxer` implementing `oxideav_core::Demuxer`. Parses the prelude up to
  the `movi` opener, exposes two streams (video `mjpeg` at `1/fps`, audio
  `adpcm_amv` at `1/samples_per_sec`), and walks `movi` chunk-by-chunk
  emitting one packet per `00dc` / `01wb` leaf. Terminates at the
  `AMV_END_` 8-byte ASCII trailer with `Error::Eof`.
- Audio packet PTS / duration derived from the 8-byte preamble's
  `decoded_sample_count` field (§4b) so downstream pipelines can advance
  the audio clock by exact decoded-sample counts.
- Probe + registry wiring — `oxideav_amv::probe` returns `100` on the
  `RIFF .... 'AMV '` 12-byte signature, and `register` installs the demuxer
  under name `amv` with the `.amv` extension hint.
- End-to-end fixture test against `docs/container/amv/fixtures/comedian.amv`
  (1116 video + 1116 audio chunks, 128 × 96 @ 12 fps, 1:33 packed duration,
  first video chunk size 1633 bytes).

### Changed

- Crate description updated to reflect the demuxer-only scope (was: codec
  scaffold).
- `Cargo.toml` no longer depends on `oxideav-mjpeg` — the container does
  not run the JPEG decoder itself; downstream consumers wire that in.

## [0.0.8] — 2026-05-18

### Erased

- Prior master history was force-erased on **2026-05-18** under
  Hat-3 cold enforcement of the workspace clean-room policy
  (`docs/IMPLEMENTOR_ROUND.md`).

### Reset

- Crate reduced to a minimal `oxideav_core::register!` stub. Every
  public API returns `Error::NotImplemented`.
