# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `AmvDemuxer::seek_to` — linear-walk seek over the `movi` payload.
  AMV carries no `idx1` / OpenDML index (trace §1 quirk #2), so the
  implementation rewinds to `movi_start` when the target PTS is behind
  the current cursor and walks forward otherwise. Chunk bodies are
  skipped via `Seek` so video JPEGs are never allocated on the seek
  path; the 8-byte §4b audio preamble is read inline to keep the
  running decoded-sample-count up to date. Every video chunk is intra
  (§4a) so the stream-0 landing is exact; stream-1 lands at the first
  chunk whose cumulative sample count reaches or exceeds the request.
  Six unit tests cover backwards-rewind, forward-walk, audio
  cumulative-PTS, past-end clamping, PTS-zero rewind, and argument
  validation; one real-fixture test (`comedian_fixture_seek_to_video_frame_500`)
  drains to EOF, rewinds to video frame 500, and confirms the recovered
  payload still starts with `FF D8` (JPEG SOI per §4a).
- `AmvMuxer` — byte-faithful inverse of `AmvDemuxer`. Writes a complete
  AMV container from a `[video, audio]` `StreamInfo` pair: the §1 zeroed
  RIFF / LIST sizes, the populated §2 `amvh` body (packed-byte duration
  patched in `write_trailer` from the observed video-frame count), the
  §3 all-zero stream-header bodies plus the 20-byte audio `WAVEFORMATEX`,
  the §4 no-byte-padding chunk walk (`00dc` / `01wb` tagged by
  `Packet::stream_index`), and the §4c `AMV_END_` ASCII trailer. The
  muxer is registered under name `amv` alongside the demuxer.
- Round-trip test suite — mux 3 (video, audio) pairs through `AmvMuxer`,
  parse the result through `AmvDemuxer`, recover byte-identical payloads;
  separately, mux 1116 frames at 12 fps and confirm the patched duration
  decodes to the §2 worked-example `1:33`. A second test pins the real
  boxed `AmvMuxer` output against the test-only `build_via_no_box_muxer`
  helper so both code paths emit identical bytes.

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
