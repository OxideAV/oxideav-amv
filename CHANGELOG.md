# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- §4 strict 1:1 video-first interleave validator — new
  `validate_movi_interleave` free function (also re-exported at the
  crate root) that walks a `&[ChunkKind]` sequence and confirms the
  trace doc's §4 invariants: even-indexed chunks carry `00dc` video
  frames, odd-indexed chunks carry `01wb` audio blocks, the total
  count is even, and no chunk falls outside the observed
  `00dc` / `01wb` tag set. Per trace §4 every observed `.amv` file's
  `movi` payload follows this rigid 1:1 pairing rule —
  `comedian.amv` carries `1116 + 1116 = 2232` chunks,
  `noel-son-lumiere.amv` carries `2928 + 2928 = 5856` chunks, and
  the byte walk advances `8 + size` per chunk for the entire
  sequence under the rigid pattern. Failures report the first
  offending chunk position and the structural rule that was
  violated (audio-first first chunk, two consecutive videos / two
  consecutive audios, trailing unpaired video, or an out-of-set
  tag with the four tag bytes echoed). Useful for tooling that
  wants to confirm a recovered / extracted chunk sequence follows
  the device profile's strict pairing rule — for example, to flag
  a truncated file whose final video chunk has no following audio
  block, or a corrupted file with an unexpected non-`00dc`/`01wb`
  tag inside `movi`. Nine new unit tests cover the empty-slice
  accept, the smallest-valid-pair accept, the three-pair
  alternation accept, the audio-first reject naming `#0`, the
  consecutive-videos reject naming `#1`, the consecutive-audios
  reject naming `#2`, the trailing-unpaired-video reject naming
  the missing-trailing-audio condition, the `Other(_)` tag reject
  echoing the four tag bytes, and a worked-example
  `comedian.amv`-size 2232-chunk accept that mirrors the trace's
  end-to-end alternation count exactly. The demuxer's hot path
  does not invoke this check (it forwards chunks as packets per
  the container's no-decode contract); strictness is opt-in.

- `cargo-fuzz` harness under [`fuzz/`](./fuzz/) with two panic-free
  targets covering the parse + demuxer-open public surface.
  - `parse` exercises every public byte parser (`AmvHeader::parse`,
    `AmvWaveFormat::parse`, `ChunkHeader::parse`,
    `AmvAudioPreamble::parse`, `AmvDuration::from_packed` /
    `from_frame_count`) and every strict / cross-check helper
    (`AmvHeader::validate_sentinels`,
    `AmvWaveFormat::validate_sentinels`,
    `AmvWaveFormat::frame_interval_samples`,
    `AmvAudioPreamble::validate_sentinels`,
    `AmvAudioPreamble::is_consistent_with_frame_interval`,
    `AmvDuration::is_consistent_with_frame_count`) plus the §4a
    video-payload validators (`validate_video_payload_shape`,
    `validate_video_payload_no_internal_markers`) against
    arbitrary fuzz-supplied bytes.
  - `demuxer_open` drives the full `AmvDemuxer::open` + bounded
    `next_packet` drain path (capped at 32 packets per fuzz
    iteration) plus the strict-mode `AmvDemuxer::open_strict`
    variant, exercising the §1 RIFF probe, the §2 / §3 prelude
    walk, the §4 `movi` chunk loop, the §4c `AMV_END_` trailer
    detection, and the device-cut truncation-recovery surface
    against arbitrary attacker-controlled inputs.
  - The contract under test is that every entry point returns a
    typed `Result<…, AmvDemuxerError>` (or `oxideav_core::Result<…>`
    for the demuxer-open path) for any input byte sequence — no
    panic, no integer overflow in a debug build, no out-of-bounds
    index, no allocation proportional to an attacker-controlled
    `size` field in a chunk header. Both targets build clean on
    `nightly-aarch64-apple-darwin` via
    `cargo fuzz build {parse,demuxer_open}` and survive a 2 000
    mutated-iteration smoke run plus the staged `comedian.amv`
    device fixture as a seed input. The fuzz crate carries
    `default-features` on (matching the demuxer's open path) and
    pulls `libfuzzer-sys = "0.4"` alongside `oxideav-core = "0.1"`
    + the sibling `path = ".."` reference. The fuzz subdirectory
    is excluded from the parent crate's compilation via its own
    `[workspace] members = ["."]` declaration so the umbrella
    workspace does not pull it into normal `cargo build`.

- §4b ↔ §3b ↔ §2 frame-interval cross-check helpers — new
  `AmvWaveFormat::frame_interval_samples(fps) -> u32` typed accessor
  exposes the trace §4b worked-example sample budget per frame interval
  (`nSamplesPerSec ÷ fps`, integer truncation), reproducing the trace's
  recorded `22 050 ÷ 12 = 1837` mono-sample count for the comedian
  device profile's first audio block (and `22 050 ÷ 16 = 1378` for the
  noel profile). Companion
  `AmvAudioPreamble::is_consistent_with_frame_interval(samples_per_sec, fps)`
  cross-checks a parsed §4b preamble's `decoded_sample_count` against
  that derivation — `true` when the per-block sample count matches
  the integer-division budget exactly, `false` otherwise. Useful for
  tooling that wants to confirm a per-block sample count is consistent
  with the stream's frame-interval budget without re-implementing the
  §4b derivation — e.g. flagging a recovered truncated chunk whose
  preamble was clipped mid-write. Both helpers guard a zero `fps` so
  they stay infallible: `frame_interval_samples` short-circuits to
  `0`, and `is_consistent_with_frame_interval` only returns `true` on
  zero fps when the parsed sample count is also `0`. End-to-end test
  pins both helpers' arithmetic against each other across four fps
  values (12 / 16 / 24 / 30).

- §2 frame-count → packed-duration helpers — new
  `AmvDuration::from_frame_count(frame_count, fps)` constructor and
  companion `AmvDuration::is_consistent_with_frame_count(frame_count,
  fps)` cross-check, both surfaced on the existing public
  `AmvDuration` byte parser. `from_frame_count` applies the trace §2
  worked example as a pure function: `total_seconds = frame_count /
  fps`, then split into `[seconds, minutes, hours]` with each
  component saturating at `u8::MAX`. Reproduces `comedian.amv`'s
  device-written `0x0000_0121` (1116 chunks ÷ 12 fps → 1:33) exactly
  when round-tripped through `to_packed`. `is_consistent_with_frame_count`
  returns `true` when the receiver matches what `from_frame_count`
  would produce for the supplied `(frame_count, fps)` pair, and
  `false` on a mismatch — surfacing exactly the header-vs-chunk-count
  discrepancy a truncation-recovery pass needs to catch. Worked
  example: the comedian profile passes `(1116, 12)` and fails
  `(1115, 12)`; the noel profile fails `(2928, 16)` because the
  device-written header records the source clip's 3:02 duration
  (`0x0000_0302`) but the chunk count would integer-divide to 3:03
  (`0x0000_0303`). Both helpers guard a zero `fps` so they stay
  infallible without recreating the higher-level sentinel rejection.
  Nine new unit tests cover the comedian round-trip, the noel
  flooring behaviour, the hours-component roll-over, the
  zero-fps short-circuit, the `u8::MAX` hours saturation, the
  consistent-comedian accept, the noel-mismatch reject, the
  one-short-of-comedian reject, the zero-fps-only-matches-zero
  case, and the `from_frame_count`-then-`to_packed` round-trip for
  both fixtures' chunk counts. The muxer's `write_trailer`
  duration-patch path is refactored to delegate to
  `AmvDuration::from_frame_count`, so the worked-example arithmetic
  is recorded in exactly one place (the public byte-parser API) and
  the existing mux → demux round-trip test still patches in the
  expected `0x0000_0121` for the comedian fixture's 1116 frames at
  12 fps.

- §4a strict-marker video-payload sentinel — new
  `validate_video_payload_no_internal_markers` free function (also
  re-exported at the crate root) that walks the entropy window of a
  `00dc` chunk body between the SOI and EOI brackets and rejects any
  `FF xx` marker pair other than the byte-stuffing `FF 00` and the
  fill-byte `FF FF` variants. Per trace §4a the device-stripped
  bitstream carries **no** internal JPEG marker segments — no APP0
  (`FF E0`), no DQT (`FF DB`), no SOF0 (`FF C0`), no DHT (`FF C4`),
  no SOS (`FF DA`) — because the player splices its hardcoded quant /
  Huffman tables back in before decoding; the new helper is the
  strict counterpart of `validate_video_payload_shape` that confirms
  this invariant explicitly, reporting the offending marker byte and
  the byte position relative to the start of the chunk body when one
  is found. Eleven new unit tests cover the empty-entropy-window
  degenerate accept, the plain-entropy / byte-stuffed / fill-byte
  accepts, all five named-marker rejects (APP0 / DQT / SOF0 / DHT /
  SOS) with offset-position reporting, the premature-EOI reject, and
  the short-payload reject. A real-fixture
  `comedian_fixture_all_video_chunks_pass_no_internal_markers` test
  walks the staged `comedian.amv` end-to-end and asserts every one of
  the 1116 video chunks passes the new strict-marker scan, matching
  the trace's "first-frame marker scan found only SOI and EOI"
  observation across every frame in the fixture.
- §4a video-payload + §4b audio-preamble byte-shape sentinel validators
  — new `validate_video_payload_shape` free function plus
  `AmvAudioPreamble` byte-parser type with its own
  `validate_sentinels` helper. The video helper confirms the §4a
  invariants the trace records: the payload begins with `FF D8`
  (`JPEG_SOI`) at offset 0 and ends with `FF D9` (`JPEG_EOI`) at
  offset `size − 2`, naming the offending byte position when either
  invariant fails. The audio helper parses the 8-byte preamble into
  `(state, decoded_sample_count)` and gates strict validation on
  `decoded_sample_count > 0` — the one §4b cross-checkable
  invariant — while surfacing the `state` field verbatim because the
  trace records that value as per-block-varying and does not pin a
  constant. Public constants `JPEG_SOI`, `JPEG_EOI`, and
  `AMV_AUDIO_PREAMBLE_LEN` accompany the helpers so external tooling
  can reference the same byte tokens. Twelve new tests cover the
  comedian-shape SOI/EOI bracket, the 4-byte degenerate accept, the
  short-payload reject, the wrong-SOI / wrong-EOI rejections naming
  the byte position, the comedian first-block preamble shape, the
  noel first-block preamble shape, the zero-sample-count reject, and
  the "state field is not gated" invariant. A real-fixture
  `comedian_fixture_all_chunks_pass_payload_shape_sentinels` test
  walks the staged `comedian.amv` end-to-end and asserts every one
  of the 1116 video chunks passes the §4a SOI/EOI bracket check and
  every one of the 1116 audio chunks passes the §4b preamble check.
- `AmvDuration::to_packed` + `AmvDuration::total_seconds` — inverse
  round-trip helpers on the parsed `amvh +0x34` duration view.
  `to_packed` re-encodes the `[seconds, minutes, hours, 0]` little-
  endian layout the trace doc §2 records (the fourth byte is always
  written as `0` per the two observed device profiles), and
  `total_seconds` returns the duration as a whole-second `u32`
  (saturating-safe because the 8-bit-per-component packed layout
  caps the maximum duration well inside `u32::MAX`). Five new tests
  cover the comedian `0x0000_0121` and noel `0x0000_0302` round-trips,
  the always-zero +0x37 byte invariant, and the two `total_seconds`
  worked examples (1:33 = 93 s; 3:02 = 182 s).
- §3b audio `WAVEFORMATEX` device-profile sentinel validation — new
  `AmvWaveFormat::validate_sentinels` byte-parser helper plus
  integration into `AmvPrelude::parse_strict` /
  `AmvDemuxer::open_strict` so strict mode cross-checks the six fixed
  values the trace doc's §3b table records for the audio `strf` body:
  `wFormatTag == 1` (declared PCM at `+0x00`), `nChannels == 1`
  (mono at `+0x02`), `nAvgBytesPerSec == nSamplesPerSec * 2` (the
  decoded-PCM rate derivation at `+0x08`, matching the observed
  `44_100 == 22_050 * 2`), `nBlockAlign == 2` (16-bit block-align at
  `+0x0C`), `wBitsPerSample == 16` (decoded-PCM width at `+0x0E`), and
  `cbSize == 0` (WAVEFORMATEX extension marker at `+0x10`).
  `nSamplesPerSec` itself stays unvalidated — the trace records only
  the one observation (22 050 Hz) and does not document whether other
  Actions / ALi-chip device profiles also vary the audio sample rate.
  Eleven new tests cover acceptance of the comedian profile, a
  hypothetical 44 100 Hz cross-rate (validating the rate-free design),
  six per-constant rejections naming each `+0xNN` offset, two
  prelude-level integration rejections (non-PCM `wFormatTag` and
  inconsistent `nAvgBytesPerSec`), a demuxer-level
  `open_strict_rejects_wrong_audio_block_align` permissive-versus-strict
  divergence assertion, and an extended real-fixture cross-check
  confirming the staged `comedian.amv` device file satisfies every
  §3b invariant in addition to the previously-validated §2 / §3 set.
- §2 + §3 strict-mode sentinel validation — new `AmvDemuxer::open_strict`
  entrypoint plus `AmvHeader::validate_sentinels` byte-parser helper
  that cross-checks the trace doc's §2 (`amvh` body sentinel constants)
  and §3 (all-zero stream-header bodies) invariants the permissive
  parser silently accepts. §2 covers three relationships fixed by the
  doc table: `dwMicroSecPerFrame == 1_000_000 / fps` (the `+0x00 ÷
  +0x28` derivation that holds 83 333 / 12 in `comedian.amv` and
  62 500 / 16 in `noel-son-lumiere.amv`), `flag_one == 1` at the
  `+0x2C` constant, and `reserved_30 == 0` at `+0x30`. §3 covers the
  four "entirely zero" stream-header bodies the trace observes —
  56-byte video `strh`, 36-byte video `strf` (no `BITMAPINFOHEADER`),
  48-byte audio `strh` — and any non-zero byte in those regions
  surfaces an `InvalidData` error naming the offending file offset.
  The default `AmvDemuxer::open` entrypoint stays permissive so
  existing callers keep their relaxed acceptance; strictness is
  opt-in. Fifteen new unit tests cover the §2 sentinel triple under
  both the comedian (12 fps) and noel (16 fps) device profiles, four
  rejection shapes (zero fps, inconsistent micros_per_frame, wrong
  flag_one, non-zero reserved_30), a §3 video-strh non-zero rejection,
  a permissive-versus-strict divergence assertion against a corrupted
  `reserved_30` byte, three demuxer-level strict-open shapes (synthetic
  comedian accept, corrupted flag_one reject, non-zero video-strh
  reject), and a real-fixture cross-check confirming the staged
  `comedian.amv` device file satisfies every §2 + §3 invariant.
- §4c trailer-recovery — `AmvDemuxer` now drains gracefully when a
  `.amv` file is truncated short of the `AMV_END_` ASCII trailer
  (common in field-collected files from cheap portable players that
  lose power mid-write). Any short read at a chunk-header boundary,
  inside a chunk header, or inside a chunk body is treated as a
  graceful EOF — every complete chunk preceding the truncation is
  still emitted as a normal `Packet`, and the new
  `AmvDemuxer::is_truncated()` accessor returns `true` so callers
  can tell apart "drained 1116 / 1116 chunks via the trailer" from
  "drained 1043 / 1116 chunks then the device died". The flag stays
  `false` while the walker is still inside the payload and only flips
  on the truncation-driven EOF, never preemptively. `build_chunk_index`
  applies the same recovery: a truncated tail (missing trailer, partial
  chunk header, partial audio preamble) breaks the build cleanly
  instead of returning an error, and the resulting index covers every
  chunk that did fully land — so an indexed seek built post-truncation
  still navigates the surviving payload correctly. Ten new tests cover
  the complete-file baseline, missing-trailer, mid-chunk-header,
  mid-video-body, mid-audio-body, zero-bytes-after-last-chunk, the
  flag-doesn't-flip-mid-walk invariant, two index-build truncation
  patterns, and post-truncation indexed seek.
- `AmvDemuxer::build_chunk_index` + `AmvDemuxer::chunk_index` —
  lazy in-memory chunk-index cache. AMV files carry no embedded index
  (trace §1 quirk #2), so the build walks the `movi` payload once and
  records every chunk's file offset, kind, and pre-emit per-stream PTS
  values into a `Vec<ChunkIndexEntry>` (newly-exported public type).
  Once populated, `seek_to` switches from its disk-walking loop to a
  binary-search-style lookup over the cached entries — no more
  re-reading every chunk header per seek. The build is idempotent and
  preserves the walker's current cursor / PTS counters so it can be
  invoked mid-walk. Nine new unit tests cover index population,
  chunk-order parity, walker-state preservation, indexed-vs-linear
  seek parity across multiple PTS values, past-end clamping,
  idempotency, and a real-fixture indexed seek that confirms the
  recovered `comedian.amv` frame 500 still starts with the JPEG SOI
  bytes (`FF D8`).
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
