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

This is a **container** crate that additionally **encodes and decodes** the
AMV device's intrinsic, fixed video and audio profiles end-to-end. AMV's
video is a table-*stripped* JPEG whose quant / Huffman tables, geometry and
scan parameters live nowhere on the wire — they are hardcoded in the
player — so every AMV file shares one fixed device profile; coding it is
intrinsic to the format, not a generic codec carried in a generic
container. The crate therefore provides `decode_frame` (bare `00dc`
payload → upright RGB pixels) and `decode_audio_block` (nibble-packed
`01wb` body → 16-bit PCM) on the read side, their byte-inverses
`encode_frame_rgb` (RGB → bare `00dc` baseline JPEG) and
`encode_audio_payload` (16-bit PCM → `01wb` IMA-ADPCM block) on the write
side, and the wire-format helper `reconstruct_jpeg` (which splices the
stripped JPEG markers back to a conforming JFIF/JPEG for a generic
downstream decoder). All are pure trace-derived realisations of the device
profile; a real AMV file round-trips decode → encode → mux → demux →
decode.

## Capabilities

### Demuxer

- Parses the `amvh` main header (resolution, fps, packed duration) and
  the audio `WAVEFORMATEX`, walks the `movi` payload, and emits one
  `oxideav_core::Packet` per `00dc` (video) / `01wb` (audio) leaf chunk,
  terminating cleanly at the `AMV_END_` trailer. The stream
  `CodecParameters` describe the *decode output*: the video stream is
  `Yuv420P` at the `amvh` resolution, and the audio stream is 16-bit
  signed (`SampleFormat::S16`) mono — what the in-crate `decode_*`
  paths emit — even though the on-disk video JPEG is table-stripped and
  the on-disk audio payload is 4-bit ADPCM nibbles.
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

### In-crate video frame decode

`decode_frame(&AmvVideoFrame)` (and the convenience
`decode_frame_from_payload(&AmvHeader, payload)`) decodes a bare `00dc`
frame straight to an upright RGB `DecodedFrame { width, height, rgb }` —
no external binary, no synthesised intermediate JPEG. It is a from-scratch
baseline-JPEG decoder over the §4a device profile: a byte-stuffing-aware
MSB bit reader over the entropy window, a canonical T.81 Huffman walk
built from the Annex K K.3/K.4 `BITS`/`HUFFVAL` lists, zig-zag dequant
against K.1/K.2, a separable 8×8 inverse DCT, the 4:2:0 MCU layout (2×2
luma + 1 Cb + 1 Cr), nearest-neighbour chroma upsampling, BT.601
YCbCr→RGB, and the §4a bottom-up vertical flip so the output is upright.
The same Annex K table constants back both this decoder and
`reconstruct_jpeg` (shared in one place, not duplicated).
`AmvDemuxer::decode_video_packet(&packet)` is the demux→pixels one-call
convenience: it binds the demuxer's parsed `amvh` geometry to a video
`Packet`'s raw `00dc` payload and returns the upright `DecodedFrame`
(rejecting a non-video packet).

An **end-to-end decode-to-pixels** integration test
(`tests/decode_to_pixels.rs`) validates this against a **black-box JPEG
decoder binary** (`djpeg` from libjpeg, falling back to `magick`): the
in-crate decode of real `comedian.amv` frames matches the reference
decoder's pixels within **MAE ≈ 1.35/channel** (the only divergence is the
reference's integer fast-IDCT + fancy chroma upsampling vs the in-crate
float IDCT + nearest upsample; a wrong table / sampling / colour path is
tens of levels off — the mis-oriented baseline is ~18). A second test
decodes **all 1116 frames** in-crate with no external binary, asserting
each is 128 × 96 and that the stream is overwhelmingly coherent natural
content (the occasional genuinely-flat fade frame is accepted only when
near-perfectly uniform, never noisy). The reference tests skip when no
decoder binary is on `PATH`; no decoder *source* is ever read — the
validator is an opaque process.

A **synthetic-geometry** unit harness hardens the trace §4a *"behaviour
for non-multiple-of-16 dimensions is untested here"* gap without any
external fixture or JPEG encoder: it hand-builds a bare `00dc` entropy
stream from the public Annex K Huffman codes the decoder already uses
(MSB-first bit writer with `FF`→`FF 00` re-stuffing) and decodes it back
through `decode_frame_from_payload`. A uniform-DC stream must crop to a
flat raster at a spread of non-mod-16 geometries (17×17, 20×12, 33×9,
1×1, 96×65) — proving no padding leaks from the 16×16-MCU-aligned planes
into the W×H crop — and a per-MCU DC ramp (each MCU bumps the luma
predictor `+1`, fingerprinting its raster index into a flat tile) is
checked pixel-for-pixel against the expected MCU→pixel mapping, pinning
the crop *position* and the §4a bottom-up flip at widths/heights that
cross MCU boundaries mid-tile. The comedian fixture (128×96, both mod-16)
cannot exercise this path; the synthetic harness is the geometry-coverage
complement to the real-frame reference cross-check.

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

`decode_audio_payload(payload)` is the whole-payload convenience (the
audio counterpart of `reconstruct_jpeg_from_payload`): it takes a full
`01wb` leaf-chunk body — the 8-byte §4b preamble plus the compressed
nibbles — parses the preamble and decodes the rest in one call.
`AmvDemuxer::decode_audio_packet(&packet)` is the demux→PCM one-call
convenience and the audio mirror of `decode_video_packet`: it runs that
decode over an audio `Packet`'s raw `01wb` payload and returns the
mono 16-bit samples (rejecting a non-audio packet). An
**end-to-end PCM** integration test (`tests/decode_audio_pcm.rs`) drives
the whole `comedian.amv` audio track through it into one 2 050 650-sample
mono buffer, wraps it in a standard WAV, and cross-checks it with a
**black-box `ffprobe`**, which independently reads back 22 050 Hz, mono,
93.000 s. The probe is opaque (no audio-tool source read); the test skips
when `ffprobe` is absent.

### Encoder (video + audio)

The crate is also the device-table-locked **write side**, the byte-inverse
of the two decode paths. `encode_frame_rgb(width, height, rgb)` (and the
`encode_frame(&DecodedFrame)` convenience) turns an upright RGB raster into
a bare `00dc` payload — `FF D8` + byte-stuffed entropy + `FF D9`,
table-stripped per §4a. It is a from-scratch baseline-JPEG encoder over the
§4a device profile: BT.601 RGB→YCbCr, the §4a bottom-up DIB coded order
(inverse of the decoder's flip), 4:2:0 chroma box-averaging, a forward 8×8
DCT (the transpose of the decoder's IDCT), round-to-nearest quant against
Annex K K.1/K.2, and canonical T.81 Huffman *encode* tables built from the
same K.3/K.4 `BITS`/`HUFFVAL` lists the decoder walks (shared from
`jpeg_reconstruct`, not duplicated). Edge pixels replicate into the 16×16
MCU pad so non-multiple-of-16 geometry codes cleanly. The output passes the
decoder's strict §4a no-internal-markers bind, and encode∘decode is a
stable JPEG fixed point.

`encode_audio_payload(samples)` (and the `encode_audio_nibbles(samples)`
split that returns the predictor seed + nibble body) turns 16-bit mono PCM
into a complete `01wb` payload — the 8-byte §4b preamble plus the
nibble-packed IMA-ADPCM body, the inverse of `decode_audio_payload`. The
standard IMA/DVI forward step tracks the decoder's *reconstructed*
predictor so the two never drift; per §4b the first sample becomes the
header `int16` seed, the `+0x02` `initialStepIndex` is emitted as `0` (the
decode-verified reset value), and nibbles pack low-first. Decode∘encode is
the canonical IMA fixed point — byte-idempotent on real device blocks.

An **end-to-end encoder round-trip** (`tests/encode_roundtrip.rs`) proves
the full loop the encoder was built for: a real `comedian.amv` is decoded,
each frame / block re-encoded, re-muxed through `AmvMuxer`, then re-demuxed
and re-decoded entirely through the crate's public surface (no external
binary). The re-muxed file is a valid AMV (zeroed RIFF sizes, no-padding
chunk walk, `AMV_END_` trailer, §2 1:33 duration), the re-demux recovers
1116/1116 paired chunks under the §4 strict interleave, video tracks the
source decode at MAE < 3/channel (globally stable — the float DCT is not
bit-exact on real high-frequency content), audio re-decode is byte-exact,
and the loop converges across generations.

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
