# oxideav-amv

A pure-Rust demuxer **and** muxer for the **AMV** ("Actions Media
Video") container — the non-standard AVI variant used by inexpensive
portable media players (S1 / Actions / ALi-chip devices) — built on the
[oxideav](https://github.com/OxideAV/oxideav-workspace) framework.

AMV pairs a custom intra-only Motion-JPEG-like video stream (fixed,
hardcoded quant / Huffman tables) with an IMA-ADPCM-style mono audio
stream wrapped in a `RIFF` form whose type is `AMV ` (trailing space). It
reuses the AVI 1.0 RIFF vocabulary (`LIST`, `hdrl`, `strl`, `strh`,
`strf`, `movi`, the `<nn>dc` / `<nn>wb` chunk-tag convention,
`WAVEFORMATEX`) while discarding most of its semantics: all RIFF / LIST
sizes are zeroed, leaf chunks are not padded to an even byte boundary,
stream-header bodies are blank, there is no `idx1` index, and the stream
is bounded by an `AMV_END_` ASCII trailer instead of a RIFF chunk
length. See the
[trace document](https://github.com/OxideAV/oxideav-workspace/blob/master/docs/container/amv/amv-container-trace.md)
in the workspace for the full byte-by-byte layout.

This is primarily a **container** crate — it identifies, demuxes and
muxes AMV files and declares the video stream as the `mjpeg` codec id and
the audio stream as `adpcm_amv`. It additionally carries the two
*decode-adjacent wire-format helpers* the AMV device's stripped/hardcoded
profile makes intrinsic to the format: `reconstruct_jpeg` splices the
device-stripped JPEG marker segments back into a `00dc` frame, and
`decode_audio_block` applies the standard IMA/DVI ADPCM recurrence to an
`01wb` block (see below). Both are pure trace-derived realisations of the
device profile; the heavyweight DCT/Huffman image decode remains the
downstream `mjpeg` codec's job.

## Capabilities

### Demuxer

- Parses the `amvh` main header (resolution, fps, packed duration) and
  the audio `WAVEFORMATEX`, walks the `movi` payload, and emits one
  `oxideav_core::Packet` per `00dc` (video) / `01wb` (audio) leaf chunk,
  terminating cleanly at the `AMV_END_` trailer.
- **Seeking** via `Demuxer::seek_to` — rewinds and walks forward,
  skipping chunk bodies via `Seek` (video JPEGs are never allocated on
  the seek path); intra-only video lands exactly at the requested PTS,
  audio at the first chunk whose cumulative sample count reaches it. An
  optional `build_chunk_index()` records every chunk offset + PTS so
  repeated random-access seeks short-circuit the disk walk.
- **Truncation tolerance** — short reads at any chunk boundary are
  treated as EOF; every complete preceding chunk is still emitted, and
  `is_truncated()` / `trailer_offset()` / `trailer_matches_eof()`
  classify the three terminal states (still walking, clean trailer EOF,
  truncated EOF).
- **Strict mode** — `open_strict` (and `AmvHeader::validate_sentinels` /
  `AmvWaveFormat::validate_sentinels`) gate on the device-profile
  sentinel constants from the trace before any `movi` work, including the
  §2 `amvh` reserved 7-dword span (`+0x04..+0x1C`) and the §3 all-zero
  stream-header bodies; the default `open` stays permissive.
- **Consistency checks** — `video_frames_emitted()` and
  `duration_consistent_with_drained_frames()` tie the `movi` walk back
  to the `amvh` packed-byte duration via the trace invariant
  (frames ÷ fps = duration). `audio_blocks_emitted()` and
  `movi_interleave_balanced()` add the §4 strict 1:1 video:audio pairing
  cross-check at the demuxer level: after a clean trailer-bounded drain
  the `01wb`-block count equals the `00dc`-frame count ("1116 / 1116,
  perfectly paired"), so a truncation that cut off after a trailing video
  frame surfaces as an imbalance — without buffering the whole chunk-kind
  sequence the way the free-function `validate_movi_interleave` requires.

### Muxer

`AmvMuxer` is the byte-faithful inverse: given a `[video, audio]`
`StreamInfo` pair it writes the zeroed RIFF / LIST sizes, the packed
`amvh` duration (patched in `write_trailer` from the observed frame
count), the all-zero stream-header bodies plus the 20-byte audio
`WAVEFORMATEX`, the no-byte-padding chunk walk, and the `AMV_END_`
trailer. Mux → demux round-trips byte-identically.

### JPEG header reconstruction

The AMV device's encoder *strips* the JPEG marker segments (`DQT`,
`SOF0`, `DHT`, `SOS`) from every `00dc` video frame — the on-disk
payload is `FF D8` + bare entropy-coded data + `FF D9` — and the player
splices fixed bytes back in before decode. `reconstruct_jpeg`
(and the convenience `reconstruct_jpeg_from_payload`) is that splice:
given an `AmvVideoFrame` (the §2 geometry bound to a `00dc` payload) it
emits a standards-conforming baseline JFIF/JPEG that any generic
JPEG/MJPEG decoder accepts unchanged. Per the trace §4a
reconstruction proof the inserted segments are the JPEG Annex K example
tables verbatim and unscaled — quant K.1 (luma) / K.2 (chroma) in
zig-zag order, Huffman K.3/K.4 (luma+chroma DC+AC), baseline SOF0 at the
`amvh` resolution with 4:2:0 sampling, and one full-spectral interleaved
scan. The entropy-coded bytes are copied through byte-for-byte; no DCT,
Huffman walk or dequantisation happens here (that is the codec crate's
job downstream). The fixture test reconstructs `comedian.amv`'s first
frame to a complete baseline JPEG and pins the §4a entropy head.

An **end-to-end decode-to-pixels** integration test
(`tests/decode_to_pixels.rs`) closes the loop the §4a reconstruction was
built for: it reconstructs real `comedian.amv` frames and hands them to a
**black-box JPEG decoder binary** (`djpeg` from libjpeg, falling back to
`magick`) — exactly the trace §4a reconstruction oracle. A clean decode
(the validator exits with no premature-end-of-data error) confirms the
hardcoded 4:2:0 MCU geometry matches the bit budget of the verbatim
Annex-K tables, and a luma-std / vertical-total-variation check confirms a
coherent natural image (frame 0 decodes 128 × 96 with luma std 34.9 and
vertical TV 10.6, matching the trace's ~8.8 4:2:0 figure; a wrong sampling
would desync or scramble). The test skips when no decoder binary is on
`PATH`; no decoder *source* is read — the validator is an opaque process.

`flip_rows_vertical(pixels, height, bytes_per_row)` is the §4a blit-time
**orientation** correction: the baseline-JPEG decode comes out vertically
mirrored (the `dc` "DIB" bottom-up row convention), and a single in-place
whole-row reversal yields the upright natural image. It is a post-decode
transform — kept out of `reconstruct_jpeg`, which must stay byte-faithful
to a standard JPEG — and works for any interleaved pixel format.

### AMV-IMA-ADPCM audio decode

`decode_audio_block(&AmvAudioPreamble, compressed_body)` is the audio
counterpart of `reconstruct_jpeg`: it turns the nibble-packed body of an
`01wb` block into the 16-bit PCM mono samples the §3b `WAVEFORMATEX`
declares. Per trace §4b the codec is **standard** IMA/DVI ADPCM — the
89-entry step-size table and the 8-entry index-adjust table
`{-1,-1,-1,-1,2,4,6,8}` are the canonical IMA tables, used unmodified;
the only AMV-specific traits are the container framing (the 8-byte
per-block header carrying an `int16` predictor seed, one block per video
frame, low-nibble-first packing). Each block is self-contained: the
predictor is re-seeded from the header `int16` and the step index is
reset to 0 at block start (no state carries across blocks). The decode
applies the trace's verbatim recurrence (`diff = step>>3` plus `step>>2`
/ `step>>1` / `step` for nibble bits 0/1/2, sign from bit 3, predictor
clamped to int16, step index clamped to `[0, 88]`) and keeps exactly
`decoded_sample_count` outputs, stopping early on a truncated body. The
fixture test decodes all 1116 blocks of `comedian.amv` to **2 050 650
mono samples = exactly 93.0 s at 22 050 Hz** (matching the §2 container
duration) with a sub-0.1 % ±32768 clip rate, the trace §4b
decode-sanity result. **Empirical correction:** trace §4b's "refined
header layout" reports preamble `+0x02` as "always `00 00`", but it is
non-zero in some blocks (e.g. `30` at audio block 50); the validated
decode resets the step index to 0 regardless, as the trace's gap note
("treating header +2 as the step index made the output worse")
prescribes — feeding `+0x02` in instead inflates the clip rate ~27×.

### Standalone byte parsers

The byte-level types are usable without the framework: `AmvHeader`,
`AmvDuration`, `AmvWaveFormat`, `AmvAudioPreamble`, `AmvVideoFrame`,
`ChunkHeader`, `ChunkKind`, the `MoviPayloadIter` chunk walker, the
chunk-tag / trailer / JPEG-marker constants, and free-function
validators for video-payload shape, internal-marker scanning, and the
1:1 video-audio `movi` interleave rule. `AmvVideoFrame::bind` makes the
header-to-payload dependency explicit (the `00dc` bitstream carries no
frame header — resolution comes from `amvh`), exposing the entropy-coded
window between SOI and EOI for a future decoder. AMV's per-block audio
preamble (`state`, `decoded_sample_count`) is surfaced verbatim, with
nibble-budget / sample-interval cross-checks for recovery tooling. The §4b
"refined" split re-reads the raw `state` dword as the two packed signed-16-bit
fields it actually carries — `initial_predictor()` (the per-block ADPCM seed at
`+0x00`) and `initial_step_index()` (`+0x02`, always 0 in both fixtures) — with
`step_index_in_ima_range()` range-checking the step index against the canonical
IMA `[0, 88]` table bound (`IMA_STEP_INDEX_MAX`); the nibble-to-PCM decode
itself stays the downstream `adpcm_amv` codec's job.

### Fuzz + bench

A `cargo-fuzz` harness under [`fuzz/`](./fuzz/) drives every public byte
parser and the full `open` + drain path for panic-free behaviour on
arbitrary input. A Criterion suite under [`benches/`](./benches/) A/Bs
the demux drain, index build, indexed-vs-linear seek, and mux write hot
paths, synthesising all inputs through the public muxer (no committed
fixtures).

## Public API

```rust
use std::fs::File;
use std::io::BufReader;
use oxideav_amv::AmvDemuxer;

let f = File::open("sample.amv").unwrap();
let mut demuxer = AmvDemuxer::open(BufReader::new(f)).unwrap();
println!("{}×{} @ {} fps", demuxer.header().width, demuxer.header().height,
         demuxer.header().fps);
println!("audio: {} Hz", demuxer.audio_format().samples_per_sec);
```

When wired into a `RuntimeContext` via `oxideav_amv::register`, both the
demuxer and muxer register under the container name `amv` — FORM-type
probing + `.amv` extension hint for reads, `open_muxer("amv", …)` for
writes.

## Provenance

Every byte-format fact is derived from
`docs/container/amv/amv-container-trace.md`, reverse-engineered from
observed bytes of redistributable `.amv` samples. No external
multimedia-library source code or any third-party AMV demuxer was
consulted. Field names such as `amvh`, `strh`, `strf`, `movi`, `00dc`,
`01wb` are the literal four-byte ASCII identifiers observed in the file;
their structural roles follow Microsoft's published RIFF / AVI
references staged in
[`container/riff/`](https://github.com/OxideAV/oxideav-workspace/blob/master/docs/container/riff/).

## License

MIT — see [LICENSE](./LICENSE).
