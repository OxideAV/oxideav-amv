# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
