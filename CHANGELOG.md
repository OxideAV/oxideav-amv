# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the crate adheres
to [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- §4a end-to-end **decode-to-pixels** validation — a new integration test
  (`tests/decode_to_pixels.rs`) proves the milestone the §4a JPEG-marker
  reconstruction was built for: a real `comedian.amv` `00dc` frame, run
  through `reconstruct_jpeg_from_payload`, decodes to a *coherent pixel
  raster*. The crate stays container-only (no DCT/Huffman decode lives here);
  the reconstructed standards-conforming baseline JPEG is handed to a
  **black-box JPEG decoder binary** (`djpeg` from libjpeg, falling back to
  `magick`) exactly as the trace §4a reconstruction oracle prescribes — clean
  decode with no premature-end-of-data error (the decoder exits 0, confirming
  the hardcoded 4:2:0 MCU geometry matches the bit budget of the verbatim
  Annex-K tables) plus a coherent-natural-image check (luma std and vertical
  total-variation thresholds). The three first frames decode to 128 × 96
  rasters with luma std 34.9 and vertical TV 10.6 — matching the trace's
  ~8.8 4:2:0 figure, where a wrong 4:1:1 sampling or wrong tables would desync
  or scramble. The test skips automatically when no JPEG decoder binary is on
  `PATH`. No decoder *source* is read; the validator is an opaque process.

- §4a **bottom-up orientation** helper — new `flip_rows_vertical(pixels,
  height, bytes_per_row)` (re-exported at the crate root) is the documented
  blit-time correction for the §4a orientation note: *"the decoded raster
  comes out vertically mirrored; a single vertical flip yields the upright
  natural image … consistent with the `dc` ('DIB') chunk convention (bottom-up
  DIB row order)."* It reverses whole rows in place for any interleaved pixel
  format (RGB/RGBA/grayscale/packed YCbCr), so a consumer that decodes the
  reconstructed JPEG applies it once to get the upright image. Kept out of
  `reconstruct_jpeg` (which must stay byte-faithful to a standard JPEG) — it is
  a post-decode transform, not a codec table. Six new unit tests pin the
  exact top↔bottom row swap, the involution (double-flip identity), odd-height
  middle-row preservation, single-row / empty no-ops, and the geometry-mismatch
  panic; the decode-to-pixels integration test additionally applies it to a
  *real* decoded raster and pins that the upright top row equals the mirrored
  bottom row and that a second flip restores the decoder output.

- §4b whole-payload audio decode + **PCM end-to-end** validation — new
  `decode_audio_payload(payload) -> Result<Vec<i16>, AmvDemuxerError>`
  (re-exported at the crate root) is the audio counterpart of
  `reconstruct_jpeg_from_payload`: it takes a full `01wb` leaf-chunk payload
  (the §4b 8-byte preamble **plus** body), parses the preamble off the front
  and decodes the rest, removing the manual `AmvAudioPreamble::parse` +
  `&payload[AMV_AUDIO_PREAMBLE_LEN..]` slice every consumer would repeat. A
  new integration test (`tests/decode_audio_pcm.rs`) drives the whole
  `comedian.amv` audio track through it into one contiguous 16-bit mono PCM
  buffer (2 050 650 samples), wraps it in a standard WAV, and cross-checks it
  with a **black-box `ffprobe`** — which independently reads back 22 050 Hz,
  mono, **93.000 s** (matching the §2 container 1:33). The probe is an opaque
  validator; no audio-tool source is read, and the test skips when `ffprobe`
  is absent. Three new unit tests pin the convenience against the manual split,
  the preamble-only empty case, and the short-preamble rejection.

- §4b AMV-IMA-ADPCM audio decode — new `decode_audio_block(&AmvAudioPreamble,
  compressed_body) -> Vec<i16>` (re-exported at the crate root) turns the
  nibble-packed body of an `01wb` block into the 16-bit PCM mono samples the
  §3b `WAVEFORMATEX` declares — the audio counterpart of the §4a
  `reconstruct_jpeg` wire-format helper. Per the freshly-staged trace §4b
  "IMA step / index tables — STANDARD" subsection the codec is **standard**
  IMA/DVI ADPCM: the 89-entry step-size table (7, 8, 9, … 27086, 29794, 32767)
  and the 8-entry index-adjust table `{-1,-1,-1,-1,2,4,6,8}` are the canonical
  IMA tables, used unmodified (carried as `const` arrays since `docs/` is not
  part of the published crate). The decode applies the trace's verbatim
  recurrence — `diff = step>>3`, plus `step>>2` / `step>>1` / `step` for nibble
  bits 0 / 1 / 2, sign from bit 3, predictor clamped to int16, step index
  clamped to `[0, 88]` — unpacks nibbles **low-nibble-first** (standard IMA
  byte order), and keeps exactly `decoded_sample_count` outputs (stopping early
  on a truncated body, matching the demuxer's §4c truncation tolerance). Each
  block is **self-contained**: the predictor is re-seeded from the header
  `int16` (`AmvAudioPreamble::initial_predictor`) and the step index is reset to
  0 at block start (no state carries across blocks). A real-fixture test decodes
  all 1116 blocks of `comedian.amv` to **2 050 650 mono samples = exactly
  93.0 s at 22 050 Hz** (matching the §2 container duration, 1:33) with a
  sub-0.1 % ±32768 clip rate — the trace §4b decode-sanity result (0.024 %; a
  wrong decode runs at 0.6 %+). **Empirical correction to trace §4b:** the
  "refined header layout" reports preamble `+0x02` as "always `00 00`", but it
  is in fact non-zero in some blocks of the staged fixture (e.g. `30` at audio
  block 50). The validated decode resets the step index to 0 regardless — as
  the trace's own §4b gap note prescribes ("treating header +2 as the step
  index made the output worse") — and a regression test pins that an
  `initial_step_index() == 30` preamble still decodes its first nibble at step
  index 0; feeding `+0x02` in instead inflates the fixture clip rate ~27×
  (0.024 % → 0.64 %). Twelve new unit tests cover the canonical-IMA table
  values, the per-nibble recurrence worked example, the index-never-negative
  and index/predictor clamp bounds, low-nibble-first ordering, output
  truncation to `decoded_sample_count`, short-body / empty-body handling,
  predictor re-seeding, the step-index-0-reset correction, and the full
  fixture decode-sanity pin. Decode-adjacent wire glue only — the heavyweight
  image DCT/Huffman decode stays the downstream `mjpeg` codec's job. Suggested
  docs erratum: §4b "8-byte block header layout (refined)" should soften
  "always `00 00`" for `+0x02` to "0 in the first blocks surveyed; non-zero in
  later blocks (e.g. 30); the validated decode resets the step index to 0
  regardless".

- §4b refined audio-preamble split — new `AmvAudioPreamble::initial_predictor()
  -> i16`, `initial_step_index() -> i16`, and `step_index_in_ima_range() ->
  bool` accessors surface the trace's §4b "8-byte block header layout
  (refined)" subsection. That refinement re-reads the first dword the existing
  raw `state: u32` field carries verbatim and establishes it is **not** a block
  index — "values for blocks 0–7 are `0, 1, -9, 8, 1, -2, -4, 5`
  (non-monotonic)" — but two packed signed 16-bit fields: an `initialPredictor`
  ADPCM seed at preamble `+0x00` and an `initialStepIndex` at `+0x02`
  ("always `00 00`" in both staged fixtures; each block re-seeds the predictor
  and resets the step index to 0, no state carries across blocks).
  `initial_predictor()` returns the low 16 bits of `state` reinterpreted as a
  signed `i16` (little-endian byte order preserved); `initial_step_index()`
  returns the high 16 bits. `step_index_in_ima_range()` range-checks the step
  index against the canonical IMA/DVI-ADPCM `[0, 88]` table bound (the §4b
  "89-entry step-size table … index clamped to `[0, 88]`"), exposed as the new
  public `IMA_STEP_INDEX_MAX = 88` constant; the actual IMA step/index tables
  and the nibble-to-PCM decode itself stay the downstream `adpcm_amv` codec's
  job under the container's no-decode contract. The raw `state: u32` field is
  unchanged (callers predating the refinement, or wanting the whole dword, are
  unaffected). Five new unit tests pin the all-zero first-block split, the
  signed-predictor worked example for blocks 0–7, a little-endian round-trip
  parsed straight off an `01wb` payload (predictor `-9`), the `[0, 88]` IMA
  range bound (0 and 88 in range; 89 and `-1` out), and the
  `IMA_STEP_INDEX_MAX == 88` constant. Container-format parse only; no decode.

- §4a JPEG header reconstruction — new `reconstruct_jpeg(&AmvVideoFrame)`
  and convenience `reconstruct_jpeg_from_payload(&AmvHeader, &[u8])`
  splice the device-stripped marker segments back into a bare `00dc`
  video frame, producing a standards-conforming baseline JFIF/JPEG that a
  generic JPEG/MJPEG decoder accepts unchanged. The AMV encoder removes
  the `DQT` / `SOF0` / `DHT` / `SOS` segments from every frame (on disk a
  `00dc` payload is `FF D8` + bare entropy data + `FF D9`); per the trace
  §4a reconstruction proof the player hardcodes the JPEG Annex K example
  tables verbatim and unscaled, so the reconstructor inserts: a `DQT`
  with quant K.1 (luma `Tq=0`) / K.2 (chroma `Tq=1`) emitted in zig-zag
  order (T.81 §B.2.4.1); a baseline `SOF0` at the §2 `amvh` resolution,
  8-bit, 3-component, **4:2:0** sampling (luma `2×2`, chroma `1×1`); a
  single `DHT` carrying Huffman K.3/K.4 luma+chroma DC+AC with class /
  destination `00 10 01 11`; and one interleaved `SOS` with full spectral
  selection `Ss=0 Se=63 Ah=0 Al=0`. The SOI/EOI are reused from the
  payload and the entropy-coded bytes are copied through byte-for-byte —
  no DCT, Huffman walk or dequantisation is performed (image decode stays
  the codec crate's job). All Annex K table values are the public
  ITU-T T.81 examples transcribed from the clean-room
  `docs/image/jpeg/tables/*.csv` extracts the trace §4a cites by name.
  Nine new tests cover the marker order/lengths, the zig-zag emission,
  4:2:0 SOF0 geometry, the four-table DHT, the SOS scan, verbatim entropy
  preservation, and a real-fixture reconstruction of `comedian.amv`'s
  first frame to a complete baseline JPEG pinned to the §4a entropy head.

- §4 demuxer-level 1:1 video:audio interleave cross-check — new
  `AmvDemuxer::audio_blocks_emitted() -> u64` surfaces the count of
  `01wb` audio **blocks** (chunks) the `movi` walk has drained, the
  chunk-count companion to the existing `video_frames_emitted()` (which
  the audio `pts` accumulator, a *sample* count, could not provide), and
  `AmvDemuxer::movi_interleave_balanced() -> bool` ties the two together
  via the trace's §4 strict 1:1 video-first pairing rule ("1116 `00dc`
  then 1116 `01wb`, perfectly paired"; §5 hierarchy "strict 1:1,
  video-first"). After a clean trailer-bounded drain the `01wb` and
  `00dc` counts are equal, so a truncation that cut the device off after
  a trailing `00dc` but before its paired `01wb` surfaces as an imbalance
  (`video_frames_emitted() - audio_blocks_emitted() == 1`) — the same
  unpaired-trailing-video signal the free-function
  `validate_movi_interleave` reports, but computed streaming over the
  walk **without** buffering the entire chunk-kind sequence into a `Vec`
  the way that function requires. The new counter mirrors the audio-PTS
  counter's lifecycle exactly: it is rewound / fast-forwarded with the
  cursor by `seek_to` (linear and indexed paths), preserved across
  `build_chunk_index`, and a new `ChunkIndexEntry::audio_blocks_before`
  field lets an indexed seek snap it to the target without re-walking;
  the indexed past-end case derives the count exactly (unlike the
  unrecoverable last-block sample count) so `movi_interleave_balanced`
  stays correct after a seek beyond the last chunk. Six new unit tests
  pin the streaming count, the at-pair-boundary-only `true` window, the
  full-drain balance, the truncated-trailing-video imbalance, the
  linear-seek rewind, and the indexed-seek + past-end + index-entry
  monotonic `audio_blocks_before` cases.

- §2 `amvh` reserved-span strict validation — `AmvDemuxer::open_strict`
  (via `AmvPrelude::parse_strict`) now enforces the trace's §2 "reserved
  / zeroed (7 dwords)" annotation for the 28 bytes between
  `dwMicroSecPerFrame` (+0x00) and `width` (+0x20). This was the one §2
  invariant strict mode did not cover: `AmvHeader::parse` discards that
  span (it reads only the four carry-data dwords) so `validate_sentinels`
  — which only sees the parsed struct — could not check it, leaving a gap
  alongside the §3 strh/strf all-zero checks already in `parse_strict`.
  The new check runs against the raw prelude slice (the same
  `require_all_zero` helper the §3 bodies use) and surfaces an
  `InvalidData` error naming the offending file offset. The permissive
  `AmvDemuxer::open` path is unchanged; strictness stays opt-in. Two new
  unit tests pin head (file 0x24, the first reserved dword) and tail
  (file 0x3F, the last reserved byte before `width`) corruption rejects,
  confirming the whole span is covered; the existing synthetic / fixture
  strict-accept tests continue to pass (the device profiles leave the
  span zero).

## [0.0.9](https://github.com/OxideAV/oxideav-amv/compare/v0.0.8...v0.0.9) - 2026-05-18

### Added

- §4b audio-block padding-slack accessor — new
  `AmvAudioPreamble::body_padding_slack(total_payload_len) -> Option<u64>`
  quantifies the trace's "occasionally 930" padded-block case: it returns
  `Some(0)` for the exact nibble budget (`8 + ceil(decoded_sample_count / 2)`
  = the 927-byte common case), `Some(n)` for `n` trailing padding bytes
  past that budget (the §4b 930-byte block reports `Some(3)`), and `None`
  for a payload shorter than the budget (a clipped / truncated block) or
  below the 8-byte preamble — saturating-safe, no panic, no underflow
  wrap. It is the signed companion to the boolean
  `is_consistent_with_body_len` (whose `true` set is exactly this helper's
  `Some(0)` set): a recovery / inspection pass can now tell *how* a block
  deviates — padded vs. clipped — instead of only *that* it does, and can
  recover the padded slack directly. Five new unit tests pin the exact /
  padded-930 / short / sub-preamble cases plus an agreement sweep across
  odd / even / zero sample counts against `is_consistent_with_body_len`;
  the `parse` fuzz target drives the new accessor with both the slice
  length and an attacker-chosen total length.

- §2↔§4 header / walk duration cross-check on the demuxer — new
  `AmvDemuxer::video_frames_emitted() -> u64` surfaces the count of
  stream-0 video chunks the `movi` walk has drained (the value the
  demuxer already tracks for PTS stamping, incremented by exactly one
  per `00dc` chunk), and `duration_consistent_with_drained_frames() ->
  bool` ties it back to the §2 `amvh` packed-byte duration via the
  trace's central header↔payload invariant ("1116 frames ÷ 12 fps =
  93 s = 1:33", §2). The cross-check is the demuxer-level companion to
  the existing `AmvDuration::is_consistent_with_frame_count` free
  function: that takes a caller-supplied frame count, whereas the new
  method feeds it the count the walker actually drained, so after a full
  forward drain a caller can confirm in one call that the device-written
  duration agrees with the chunks that landed. On a truncated file the
  header still records the device's intended full duration while the walk
  only reached the surviving chunks, so the cross-check returns `false`
  — the signal a recovery tool needs to decide whether to re-stamp the
  header. Four new unit tests pin the per-chunk count tracking, the
  matching (1116 frames / 1:33) and deliberately mismatched (3 frames /
  1:33) synthetic cases, and a real `comedian.amv` fixture drain that
  confirms its 1116-frame walk agrees with the packed `1:33` duration.
- §4c trailer-offset cross-check on the demuxer — new
  `AmvDemuxer::trailer_offset() -> Option<u64>` records the absolute
  file offset at which the `AMV_END_` ASCII trailer is observed when the
  `movi` walk terminates cleanly (via `next_packet` or the linear
  `seek_to` path). Because all RIFF / LIST sizes are zeroed and there is
  no `idx1` index (§1 quirks), this trailer is the *only* byte-bound the
  stream carries (§4c), and the §4 no-padding proof turns on its exact
  position. The companion `trailer_matches_eof(stream_len) ->
  Option<bool>` applies that proof: a clean `8 + size` walk lands the
  trailer at `stream_len − 8` (the trailer is the file's final 8 bytes).
  The accessor stays `None` for a stream drained by truncation (the
  trailer never landed), so together with `is_truncated()` it classifies
  all three terminal states — still walking, clean trailer EOF, or
  truncated EOF. Four new unit tests pin the clean-EOF offset and the
  `stream_len ± 1` match/mismatch, the truncation `None` case, and the
  `seek_to`-past-end recording path; the `comedian.amv` fixture walk now
  additionally pins the §4 worked example directly — trailer at
  `0x348E31`, EOF at `0x348E39`, `trailer_matches_eof` true.
- Criterion benchmark suite (depth-mode round) — four self-contained
  bench binaries under `benches/` that A/B the container hot paths, each
  synthesising its input in-bench through the crate's **public**
  `AmvMuxer` write path (no committed fixture file). `demux_drain` opens
  and walks a 1116-pair file to the §4c `AMV_END_` trailer via
  `AmvDemuxer` + `Demuxer::next_packet` (the 8-byte chunk-header parse,
  §4 no-padding `8 + size` advance, per-stream PTS accounting);
  `build_index` benches `build_chunk_index` (the Seek-skip-bodies offset
  table walk); `indexed_seek` measures a 10-target seek spread on the
  binary-search-backed `seek_to_via_index` path **and** the linear
  disk-walking fallback in one group, quantifying the index payoff
  (~7× on the local run: ~7.0 µs indexed vs ~49.4 µs linear);
  `mux_write` benches the full `write_header` + 1116-pair `write_packet`
  + `write_trailer` write path. A shared `benches/common/mod.rs` carries
  the muxer driver (an `Arc<Mutex<Cursor<Vec<u8>>>>`-backed `WriteSeek`
  so the bytes survive the muxer's `Box<dyn WriteSeek>` type erasure)
  and the `comedian.amv` `StreamInfo` pair. Bench-only; no `src/` change.
- §4a typed video-frame binding surface — new `AmvVideoFrame<'a>`
  (`src/video.rs`) binds the §2 `amvh` stream geometry to a validated
  `00dc` payload, the hand-off type a future AMV video decoder will
  consume. Per trace §4a the `00dc` bitstream carries **no** frame
  header of any kind (no quantization-table, Huffman-table, frame or
  scan marker segments — those parameters are hardcoded in the
  player's decoder) so the resolution exists *only* in the §2 header;
  the binding makes that dependency explicit and type-safe.
  `AmvVideoFrame::bind(header, body)` validates non-zero §2 geometry
  plus the §4a SOI / EOI bracket (via the existing
  `validate_video_payload_shape`); `bind_strict` additionally runs the
  §4a no-internal-markers scan. Accessors: `width()` / `height()` (§2
  body offsets +0x20 / +0x24), `body()` (full bracketed payload),
  `entropy_coded()` (the window strictly between SOI and EOI — §4a
  "immediately after SOI the bytes are entropy-coded data"), and
  `is_keyframe()` (always `true` — §4a "Every frame is a keyframe",
  intra-only). Nine new unit tests cover the minimal 4-byte bracket
  bind, entropy-window slicing against the §4a worked-example bytes,
  missing-SOI / missing-EOI / sub-minimum rejects, zero-width /
  zero-height geometry rejects, the strict-bind internal-marker
  reject vs. plain-bind accept, and byte-stuffing (`FF 00`)
  acceptance under the strict scan. A new
  `comedian_fixture_strict_binds_all_1116_video_frames` integration
  test parses the §2 `amvh` straight from the staged fixture bytes,
  walks the `movi` payload with `MoviPayloadIter`, and strict-binds
  every `00dc` chunk: 1116 frames bound, all 128 × 96, all keyframes,
  first three payload sizes pinned to the §4 chunk table
  (1633 / 1627 / 1625), and frame 0's entropy window pinned to
  `1633 − 4 = 1629` bytes beginning `E6 49 A6 93` (the §4a hexdump).
  The actual entropy decode remains out of reach of the staged trace:
  `amv-container-trace.md` §4a records that the tables / scan
  parameters are device-hardcoded but does not enumerate them.

- §4b 4-bit-per-sample nibble-budget helpers on `AmvAudioPreamble` —
  new `nibble_body_len() -> u64` returns the expected compressed-body
  byte count for the block's declared `decoded_sample_count` under the
  trace §4b nibble-packing relation (each mono sample is one 4-bit
  nibble, two per byte, so `body = ceil(decoded_sample_count / 2)`),
  reproducing the trace's recorded `ceil(1837 / 2) = 919` for the
  comedian first block. Companion
  `is_consistent_with_body_len(total_payload_len) -> bool` cross-checks
  a full `01wb` payload length against the budget — `true` when
  `total_payload_len == AMV_AUDIO_PREAMBLE_LEN + ceil(decoded_sample_count
  / 2)` (i.e. `8 + 919 = 927` for the comedian first block), `false`
  otherwise, including a saturating-safe `false` on a sub-preamble
  length rather than panicking on the underflow. Useful for a
  truncation-recovery / sanity pass that wants to flag an `01wb` block
  whose declared sample count is inconsistent with the bytes that
  actually landed — a body clipped mid-write short of the nibble
  budget its preamble promises. Eleven new unit tests cover the
  comedian first-block exact match (`1837 → 919`), even / odd / zero
  sample-count round-up cases, the full-payload `927`-byte accept, the
  one-short / one-over rejects (the relation is exact, not a lower
  bound), the sub-preamble + exactly-preamble boundary cases, and a
  `nibble_body_len`-vs-`is_consistent_with_body_len` cross-pin across
  seven sample counts. A new
  `comedian_fixture_audio_blocks_nibble_budget` integration test walks
  the staged `comedian.amv` `movi` payload, pins the first audio block
  exactly to the §4b worked example (`1837` samples / `927`-byte
  payload / `919`-byte nibble body), and asserts the majority of the
  1116 audio blocks satisfy the nibble budget — a trace-faithful loose
  lower bound that respects the doc's "927 bytes, occasionally 930"
  note (the padded outliers correctly miss the exact relation). The
  `parse` fuzz target now drives both helpers against
  attacker-controlled `decoded_sample_count` / `total_payload_len`
  pairs to confirm no overflow or panic.

- §4 typed chunk-payload iterator — new `MoviPayloadIter<'a>` walks
  an in-memory `movi`-body byte buffer (the bytes between the
  `LIST <size> 'movi'` opener and the §4c `AMV_END_` trailer) and
  yields one `MoviPayload<'a>` per chunk as `Video { chunk_offset,
  body }`, `Audio { chunk_offset, preamble, body }` (with the §4b
  8-byte preamble already parsed via `AmvAudioPreamble::parse`), or
  `Other { chunk_offset, tag, body }` for any FOURCC outside the
  observed `00dc` / `01wb` set. The iterator advances by exactly
  `8 + size` per chunk under §4's no-padding rule, terminates cleanly
  on an end-of-buffer landing, and latches a done flag after the
  first error so a malformed stream surfaces a single typed error
  rather than panicking or silently terminating mid-walk. Error
  variants cover the three §4 truncation modes observed in the trace
  doc: trailing window with `<8` bytes left for a chunk header, a
  chunk whose declared body size runs past the buffer, and an `01wb`
  payload shorter than the 8-byte preamble. The new
  `MoviPayload::kind()` accessor returns a `ChunkKind` so the typed
  iterator's output can be fed directly into the existing
  `validate_movi_interleave` validator without re-walking the bytes;
  `MoviPayload::body()` and `MoviPayload::chunk_offset()` are
  convenience accessors over the per-variant fields. Ten new unit
  tests in `parse::tests` cover the empty-buffer no-yield path, the
  single video + audio pair walk (with cursor accounting), the §4
  no-padding-on-odd-sized-payload invariant (verifying the cursor
  doesn't word-align even though both body lengths are odd), an
  unknown-FOURCC `Other` surface, trailing-truncation error +
  latched-done re-surface, declared-size overrun error, audio
  preamble shorter than `AMV_AUDIO_PREAMBLE_LEN` error, a three-pair
  walk whose collected kinds satisfy `validate_movi_interleave`, an
  `AmvAudioPreamble` round-trip match against independent parse, and
  a `body()`-accessor-matches-field cross-check. A new
  `comedian_fixture_movi_payload_iter_walks_2232_chunks` integration
  test in `demuxer::tests` loads the staged `comedian.amv` fixture,
  locates the `movi` FOURCC, slices off the §4c 8-byte `AMV_END_`
  trailer, and walks the iterator to confirm the §4 worked example
  end-to-end: 1116 + 1116 = 2232 chunks under the strict video-first
  alternation, no `Other` tags surface, the first video body is 1633
  bytes (size `0x661`), the first audio preamble carries
  `decoded_sample_count = 1837` (= `22_050 ÷ 12`), and the cursor
  lands cleanly on the end of the movi-body slice (i.e. 8 bytes
  before EOF in the original file).

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
