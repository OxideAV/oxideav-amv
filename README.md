# oxideav-amv

A pure-Rust demuxer for the **AMV** ("Actions Media Video") container — the
non-standard AVI variant used by inexpensive Chinese "MP3/MP4" portable media
players (S1 / Actions / ALi-chip devices) — built on the
[oxideav](https://github.com/OxideAV/oxideav) framework.

AMV pairs a custom intra-only Motion-JPEG-like video stream (fixed, hardcoded
quant / Huffman tables) with an IMA-ADPCM-style mono audio stream wrapped in a
`RIFF` form whose type is `AMV ` (trailing space). It reuses the AVI 1.0 RIFF
vocabulary (`LIST`, `hdrl`, `strl`, `strh`, `strf`, `movi`, the `<nn>dc` /
`<nn>wb` chunk-tag convention, `WAVEFORMATEX`) while discarding most of its
semantics — see the
[trace document](https://github.com/OxideAV/oxideav/blob/master/docs/container/amv/amv-container-trace.md)
in the workspace for the full byte-by-byte layout.

## Status

Container demuxer **and** muxer. Reads the prelude (`amvh` resolution / fps
/ duration, audio `WAVEFORMATEX`) and walks the `movi` payload, emitting
one `oxideav_core::Packet` per `00dc` (video) / `01wb` (audio) leaf chunk;
the walk terminates cleanly at the trailing `AMV_END_` ASCII literal.
Validated end-to-end against the staged `comedian.amv` fixture (128 × 96,
12 fps, 1:33 duration, 1116 + 1116 paired chunks).

The new `AmvMuxer` is the inverse: given a `[video, audio]` `StreamInfo`
pair (width/height/fps on the video side, sample-rate/channels on the
audio side), it writes a byte-faithful AMV file with the §1 zeroed RIFF
/ LIST sizes, the populated §2 `amvh` body (packed-byte duration patched
in `write_trailer` from the observed frame count), the §3 all-zero
stream-header bodies plus the 20-byte audio `WAVEFORMATEX`, the §4
no-byte-padding chunk walk, and the §4c `AMV_END_` ASCII trailer. The
test suite includes a mux → demux round-trip that recovers byte-identical
payloads and the expected `1:33` duration when fed 1116 frames at 12 fps.

Frame and sample **decoding** are out of scope — the video stream is declared
as the `mjpeg` codec id (the actual JPEG payload requires the player's stripped
quant / Huffman tables to be spliced back in, which a downstream codec
preprocessor will handle) and the audio stream as an `adpcm_amv` placeholder
codec id (reserved for the future ADPCM variant; the 8-byte per-chunk
preamble — `(state, decoded_sample_count)` per §4b — is exposed verbatim in
the packet payload).

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
demuxer and the muxer become available alongside the other registered
containers — `amv` is the container name on both sides, with FORM-type-based
probing and the `.amv` extension hint for the read path, and
`ContainerRegistry::open_muxer("amv", …)` for the write path.

The crate also exposes its byte-level parsers as a standalone library
(`AmvHeader`, `AmvDuration`, `AmvWaveFormat`, `ChunkHeader`, `ChunkKind`,
plus the chunk-tag / trailer constants) for tooling that wants to inspect
AMV files without the framework dependency at the demuxer level.

## Provenance

Every byte-format fact in this crate is derived from
`docs/container/amv/amv-container-trace.md`, which was reverse-engineered from
observed bytes of redistributable `.amv` samples. **No external multimedia-library
source code or any third-party AMV demuxer or AMV wiki was consulted.** Field
names such as `amvh`, `strh`, `strf`, `movi`, `00dc`, `01wb` are the literal
four-byte ASCII identifiers observed in the file; their structural roles follow
Microsoft's published RIFF / AVI references staged in
[`container/riff/`](https://github.com/OxideAV/oxideav/blob/master/docs/container/riff/).

## License

MIT — see [LICENSE](./LICENSE).
