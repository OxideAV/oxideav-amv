# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.9](https://github.com/OxideAV/oxideav-amv/compare/v0.0.8...v0.0.9) - 2026-05-13

### Other

- add lazy-index seek_to

### Added
- demuxer: `seek_to(stream, pts)` — AMV has no built-in chunk index, so
  the demuxer builds a lazy table of every `00dc` chunk's byte offset +
  per-stream PTS counters on first seek and binary-searches it. AMV
  video is intra-only (every frame is a keyframe) and AMV audio is
  self-contained IMA-ADPCM (per-chunk predictor + step-index header),
  so the seek lands exactly on the requested frame on the video stream
  and at the matching V-A pair on the audio stream. Overshooting clamps
  to the final indexed entry rather than erroring.

## [0.0.8](https://github.com/OxideAV/oxideav-amv/compare/v0.0.7...v0.0.8) - 2026-05-06

### Other

- drop stale REGISTRARS / with_all_features intra-doc links
- drop dead `linkme` dep
- registry calls: rename make_decoder/make_encoder → first_decoder/first_encoder
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-amv/pull/502))

## [0.0.7](https://github.com/OxideAV/oxideav-amv/compare/v0.0.6...v0.0.7) - 2026-05-03

### Other

- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- drop unused AmvAudioEncoder.sample_rate field (slim-frame leftover)
- update encode_jpeg call to slim-shape signature
- adopt slim VideoFrame/AudioFrame shape
- pin release-plz to patch-only bumps

## [0.0.6](https://github.com/OxideAV/oxideav-amv/compare/v0.0.5...v0.0.6) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core

## [0.0.5](https://github.com/OxideAV/oxideav-amv/compare/v0.0.4...v0.0.5) - 2026-04-19

### Other

- remove Cargo.lock
- bump oxideav-container dep to "0.1"
- drop Cargo.lock — this crate is a library

## [0.0.4](https://github.com/OxideAV/oxideav-amv/compare/v0.0.3...v0.0.4) - 2026-04-19

### Other

- bump container & mjpegf
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- thread &dyn CodecResolver through open()
