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

Seeking is supported via `Demuxer::seek_to(stream_index, pts)`. AMV has
no `idx1` index (trace §1 quirk #2), so the default implementation
rewinds to `movi_start` when the target PTS is behind the current cursor
and walks forward chunk-by-chunk otherwise; chunk bodies are skipped via
`Seek` so video JPEGs are never allocated on the seek path. Every video
frame is intra (§4a) so video lands exactly at the requested PTS; audio
lands at the first chunk whose cumulative §4b decoded-sample-count
reaches or exceeds the request. Validated against `comedian.amv` by
draining to EOF, rewinding to frame 500, and confirming the recovered
payload still starts with `FF D8` (JPEG SOI).

Callers that expect repeated random-access seeks can build an in-memory
chunk index up-front via `AmvDemuxer::build_chunk_index()`. The build
walks the `movi` payload once (the same Seek-skip-bodies path the seek
hot loop uses) and records every chunk's file offset plus the per-stream
pre-emit PTS into a `Vec<ChunkIndexEntry>`. Once populated, `seek_to`
short-circuits the disk-walking loop and lands directly on the matching
entry — no more re-reading every chunk header per seek. The index is
exposed read-only through `AmvDemuxer::chunk_index()` so external tools
can iterate the chunk table directly. The build is idempotent and
preserves the walker's current cursor / PTS counters so it can be
invoked mid-walk without disturbing in-flight playback.

Field-collected `.amv` files from cheap portable players regularly
arrive truncated — the user pulls the SD card mid-write, the battery
dies, the transfer is interrupted. The demuxer handles these cases
gracefully: any short read at a chunk-header boundary, mid-header, or
mid-payload is treated as EOF, every complete chunk preceding the
truncation is still emitted as a normal `Packet`, and
`AmvDemuxer::is_truncated()` returns `true` so callers can tell apart
"drained 1116 / 1116 chunks via the `AMV_END_` trailer" from "drained
1043 / 1116 chunks then the device died". `build_chunk_index` applies
the same recovery, so an index built from a truncated file covers
every chunk that did land and an indexed seek over it still navigates
the surviving payload correctly.

Tooling that wants to filter "real device-profile AMV files" from
garbled inputs up-front can use `AmvDemuxer::open_strict`. The strict
variant runs the trace doc's §2 + §3 sentinel checks before any `movi`
work: `dwMicroSecPerFrame == 1_000_000 / fps`, `flag_one == 1` at the
`+0x2C` constant, `reserved_30 == 0` at `+0x30`, and each of the four
§3 stream-header bodies (video `strh`, video `strf`, audio `strh`) is
entirely zero per the trace observation that the device writes no
`fccHandler`/`BITMAPINFOHEADER` / `auds` metadata. The same checks are
exposed at the byte-parser level via `AmvHeader::validate_sentinels()`
for callers that want to validate a parsed header in isolation. The
default `AmvDemuxer::open` entrypoint stays permissive so the existing
demuxer-open path still accepts any byte-shaped prelude that satisfies
the §1-§4 FOURCC layout — strictness is opt-in.

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
