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

Because all RIFF / LIST sizes are zeroed and there is no `idx1` index
(§1 quirks), the 8-byte `AMV_END_` ASCII trailer is the *only* byte
bound the stream carries (§4c). On a clean trailer-bounded drain,
`AmvDemuxer::trailer_offset()` reports the absolute file offset where
that literal was observed, and `trailer_matches_eof(stream_len)`
cross-checks it against a caller-supplied total length using the §4
no-padding worked example: a correct `8 + size` walk lands the trailer
at exactly `stream_len − 8`. For `comedian.amv` that is `0x348E31` —
8 bytes before the `0x348E39` EOF — and a demuxer that had mis-applied
AVI even-byte padding would desync and never see the literal at a
clean 8-byte boundary. The accessor stays `None` for a stream drained
by truncation (the trailer never landed), so together with
`is_truncated()` it classifies all three terminal states: still
walking, clean trailer EOF, or truncated EOF.

The demuxer also ties the §4 `movi` walk back to the §2 header.
`AmvDemuxer::video_frames_emitted()` reports the count of stream-0
video chunks drained so far (the value already tracked for PTS
stamping, bumped once per `00dc`), and
`duration_consistent_with_drained_frames()` confirms — after a full
forward drain — that the §2 `amvh` packed-byte duration agrees with
that count via the trace's central invariant ("1116 frames ÷ 12 fps
= 93 s = 1:33", §2). It is the demuxer-level companion to the
`AmvDuration::is_consistent_with_frame_count` free function: the
latter takes a caller-supplied count, the former feeds it the count
the walk actually produced. A clean `comedian.amv` drain reports
`true`; a truncated file returns `false` (the header records the
device's intended full duration while only the surviving chunks
landed) — exactly the signal a recovery tool needs to decide whether
to re-stamp the header via `AmvDuration::from_frame_count`.

Tooling that wants to filter "real device-profile AMV files" from
garbled inputs up-front can use `AmvDemuxer::open_strict`. The strict
variant runs the trace doc's §2 + §3 sentinel checks before any `movi`
work: `dwMicroSecPerFrame == 1_000_000 / fps`, `flag_one == 1` at the
`+0x2C` constant, `reserved_30 == 0` at `+0x30`, each of the three §3
all-zero stream-header bodies (video `strh`, video `strf`, audio
`strh`) is entirely zero per the trace observation that the device
writes no `fccHandler`/`BITMAPINFOHEADER` / `auds` metadata, **and**
the six §3b audio `WAVEFORMATEX` device-profile constants
(`wFormatTag == 1`, `nChannels == 1`,
`nAvgBytesPerSec == nSamplesPerSec * 2`, `nBlockAlign == 2`,
`wBitsPerSample == 16`, `cbSize == 0`) — leaving the
`nSamplesPerSec` rate itself free since the trace only records the
one observed value (22 050 Hz). The same checks are exposed at the
byte-parser level via `AmvHeader::validate_sentinels()` and the new
`AmvWaveFormat::validate_sentinels()` for callers that want to
validate a parsed header in isolation. The default `AmvDemuxer::open`
entrypoint stays permissive so the existing demuxer-open path still
accepts any byte-shaped prelude that satisfies the §1-§4 FOURCC
layout — strictness is opt-in.

Per-chunk payload-shape validation is also available without going
through the demuxer. `validate_video_payload_shape` confirms a `00dc`
chunk body satisfies the §4a "self-contained JPEG bracketed by
SOI..EOI" invariants — `FF D8` at offset 0, `FF D9` at offset
`size − 2` — and reports the offending byte position when either
check fails. The companion
`validate_video_payload_no_internal_markers` enforces the §4a
strict-marker invariant: the entropy window between SOI and EOI must
carry **no** internal JPEG marker segments (no APP0 / DQT / SOF0 /
DHT / SOS), only entropy-coded data with at most `FF 00` byte
stuffing or `FF FF` fill bytes. Any `FF xx` pair outside those two
exceptions is reported with the offending marker byte and its byte
position relative to the chunk-body start. `AmvAudioPreamble::parse`
decodes the 8-byte §4b preamble into a
`(state, decoded_sample_count)` view, and
`AmvAudioPreamble::validate_sentinels` gates strict validation on
`decoded_sample_count > 0` (the one trace-recorded §4b invariant the
two observed device profiles both satisfy); the `state` field is
surfaced verbatim because the trace records that field as per-block
varying state. The companion exports `JPEG_SOI`, `JPEG_EOI`, and
`AMV_AUDIO_PREAMBLE_LEN` make the same byte tokens available to
external tooling that wants to reference them directly. The staged
`comedian.amv` device file's 1116 video chunks all pass the §4a
bracket check **and** the new §4a strict-marker scan, and its 1116
audio chunks all pass the §4b preamble check (verified end-to-end by
the test suite).

On top of those validators sits `AmvVideoFrame`, the typed hand-off
surface for a future AMV video decoder. Per trace §4a the `00dc`
bitstream carries no frame header at all — quantization / Huffman
tables and scan parameters are hardcoded in the player and "the
resolution comes from `amvh` (§2)" — so a decoder can never consume a
chunk body in isolation. `AmvVideoFrame::bind(header, body)` makes the
dependency explicit: it validates non-zero §2 geometry plus the §4a
SOI / EOI bracket and yields a frame exposing `width()` / `height()`,
the full `body()`, the `entropy_coded()` window strictly between SOI
and EOI, and `is_keyframe()` (always `true` — §4a, intra-only;
`bind_strict` adds the no-internal-markers scan). The fixture test
strict-binds all 1116 `comedian.amv` frames at 128 × 96, pins the
first three payload sizes to the §4 chunk table (1633 / 1627 / 1625)
and frame 0's 1629-byte entropy window to the §4a hexdump bytes
`E6 49 A6 93`. The entropy decode itself stays out of scope until the
trace (or a follow-up doc) enumerates the device-hardcoded tables.

For tooling that wants to recompute the `amvh +0x34` packed-byte
duration independently of the muxer's `write_trailer` patch path —
for example, when re-stamping a recovered truncated file's header
after the chunk count has been determined — `AmvDuration::to_packed`
is the inverse of `AmvDuration::from_packed` and round-trips the
`[seconds, minutes, hours, 0]` layout exactly. The fourth byte is
always written as `0` per the trace doc's two observed device
profiles. `AmvDuration::total_seconds` returns the same duration as
a whole-second `u32` for callers that want the trace's worked-example
arithmetic (comedian 1:33 = 93 s, noel 3:02 = 182 s) without the µs
detour.

For tooling that wants to verify a per-block audio sample count is
consistent with the stream's frame-interval budget, the new
`AmvWaveFormat::frame_interval_samples(fps) -> u32` typed accessor
exposes the trace §4b worked-example arithmetic
(`nSamplesPerSec ÷ fps`, integer truncation) — the comedian profile
returns `1837` for 12 fps, matching the device-written first-block
sample count exactly. The companion
`AmvAudioPreamble::is_consistent_with_frame_interval(samples_per_sec,
fps)` is the cross-checking validator: it returns `true` when the
parsed §4b preamble's `decoded_sample_count` equals the integer-division
budget and `false` otherwise — surfacing exactly the per-block
sample-count discrepancy a recovered-from-truncation pass needs to
flag.

The §4b 4-bit-per-sample nibble-packing relation is exposed as a
matching pair on `AmvAudioPreamble`. `nibble_body_len() -> u64`
returns the expected compressed-body byte count for the block's
declared `decoded_sample_count` — each mono sample is one 4-bit
nibble and two nibbles pack into one byte, so
`body = ceil(decoded_sample_count / 2)`, reproducing the trace's
`ceil(1837 / 2) = 919` for the comedian first block. The companion
`is_consistent_with_body_len(total_payload_len) -> bool` cross-checks
a full `01wb` payload length against that budget: it returns `true`
when `total_payload_len == AMV_AUDIO_PREAMBLE_LEN +
ceil(decoded_sample_count / 2)` — `8 + 919 = 927` for the comedian
first block — and `false` otherwise, including a saturating-safe
`false` (no panic) on a sub-preamble length. Useful for a
truncation-recovery / sanity pass that wants to flag an `01wb` block
whose declared sample count is inconsistent with the bytes that
actually landed. The staged `comedian.amv`'s first audio block
matches the nibble budget exactly, and the majority of its 1116
audio blocks do too — a trace-faithful loose lower bound that
respects the doc's "927 bytes, occasionally 930" note, where the few
padded blocks correctly miss the exact relation. For those padded
blocks, `AmvAudioPreamble::body_padding_slack(total_payload_len) ->
Option<u64>` is the signed companion to the boolean check: it returns
`Some(0)` at the exact 927-byte budget, `Some(3)` for the §4b
"occasionally 930" padded case (three trailing bytes past the nibble
body), and `None` when the payload is shorter than the budget (a
clipped / truncated block) or below the 8-byte preamble — saturating
safe, no underflow wrap. Its `Some(0)` set is exactly the set for
which `is_consistent_with_body_len` returns `true`, so a recovery /
inspection pass can tell *how* a block deviates (padded vs. clipped)
instead of only *that* it does, and recover the padded slack directly.

The §4 strict 1:1 video-first interleave invariant — the trace's
recorded pattern that every observed `.amv` file's `movi` payload is
a flat stream of `00dc` video frames at even positions paired with
`01wb` audio blocks at odd positions, with every video paired with
exactly one audio block — is exposed as a `validate_movi_interleave`
free function over a `&[ChunkKind]` sequence. Failures name the first
offending chunk position and the rule that was violated
(audio-first first chunk, two consecutive videos or audios, trailing
unpaired video for the missing-audio-block case, or an out-of-set
tag with the four bytes echoed). Useful for tooling that wants to
confirm a recovered / extracted chunk sequence follows the device
profile's rigid pairing rule — for example, to flag a truncated
file whose final video chunk has no following audio block. The
2232-chunk pattern (`comedian.amv`'s 1116 + 1116 paired chunks)
passes end-to-end through the validator.

A typed counterpart to that validator is also available without
going through the demuxer's I/O path. `MoviPayloadIter` walks an
in-memory `movi`-body byte buffer one chunk at a time and yields a
`MoviPayload<'a>` enum that splits `Video { chunk_offset, body }`,
`Audio { chunk_offset, preamble, body }` (with the §4b 8-byte
preamble already parsed via `AmvAudioPreamble::parse`), and
`Other { chunk_offset, tag, body }` for any FOURCC outside the
observed `00dc` / `01wb` set. The iterator advances by exactly
`8 + size` per chunk under §4's no-padding rule, terminates cleanly
on an end-of-buffer landing, and latches a done flag after surfacing
the first error (truncated trailing window, declared body size that
runs past the buffer, or an `01wb` payload shorter than the 8-byte
preamble). The `comedian.amv` integration test confirms the iterator
walks all 2232 chunks (1116 video / 1116 audio) once the §4c
`AMV_END_` trailer is sliced off the input — the first video body
is 1633 bytes (size `0x661`) and the first audio preamble carries
`decoded_sample_count = 1837` (the §4b worked-example value
`22_050 ÷ 12`), exactly the trace doc's recorded constants.

`AmvDuration::from_frame_count(frame_count, fps)` applies the trace §2
worked example (`frame_count / fps → total_seconds`, then split into
`[seconds, minutes, hours]`) as a pure function, so the same
derivation the muxer's `write_trailer` patches in is also available to
standalone tooling. Components saturate at `u8::MAX` to keep the
helper infallible, and a zero `fps` short-circuits to the all-zero
duration rather than dividing by zero. The muxer's `write_trailer`
now delegates to this helper so the worked-example arithmetic lives
in exactly one place. `comedian.amv`'s 1116 video chunks at 12 fps
round-trip exactly to the device-written `0x0000_0121`;
`noel-son-lumiere.amv`'s 2928 chunks at 16 fps round to `0x0000_0303`
(3:03) — one second past the device-written `0x0000_0302` (3:02),
which carries the source clip's original duration. The companion
`AmvDuration::is_consistent_with_frame_count(frame_count, fps)` is
the cross-checking validator that compares a parsed header's
duration against an observed chunk count: it returns `true` on the
comedian (matching) pair and `false` on the noel (off-by-one) pair,
exactly the header-vs-chunk-count discrepancy a truncation-recovery
pass needs to surface.

A `cargo-fuzz` harness lives under [`fuzz/`](./fuzz/) with two
panic-free targets: `parse` drives every public byte-parser entry
point (`AmvHeader::parse`, `AmvWaveFormat::parse`, `ChunkHeader::parse`,
`AmvAudioPreamble::parse`, `AmvDuration::from_packed` /
`from_frame_count`) plus every strict / cross-check
(`validate_sentinels`, `is_consistent_with_frame_count`,
`is_consistent_with_frame_interval`) and the §4a video-payload
validators (`validate_video_payload_shape` /
`validate_video_payload_no_internal_markers`) against arbitrary
fuzz-supplied bytes; `demuxer_open` drives the full
`AmvDemuxer::open` + bounded `next_packet` drain path including the
strict-mode `open_strict` variant. The contract under test is that
every entry point returns a typed `Result<…, AmvDemuxerError>` (or
`oxideav_core::Result<…>` for the demuxer-open path) on any byte
sequence — no panic, no integer overflow in a debug build, no
out-of-bounds index, no allocation proportional to an
attacker-controlled `size` field in a chunk header. Both targets
build clean on the `aarch64-apple-darwin` nightly toolchain via
`cargo fuzz build {parse,demuxer_open}` and survive a smoke run of
2 000 mutated iterations plus the staged `comedian.amv` device
fixture as a seed input.

A Criterion benchmark suite lives under [`benches/`](./benches/) with
four self-contained binaries that A/B the container hot paths; every
bench input is synthesised in-bench through the crate's **public**
`AmvMuxer` write path, so no fixture file is committed. `demux_drain`
opens and walks a 1116-pair file (the `comedian.amv` chunk count) to
the §4c `AMV_END_` trailer through `AmvDemuxer` + `Demuxer::next_packet`
— the 8-byte chunk-header parse, the §4 no-padding `8 + size` advance,
and the per-stream PTS accounting. `build_index` benches
`build_chunk_index`, the Seek-skip-bodies walk that records every
chunk's file offset plus its pre-emit PTS. `indexed_seek` measures a
ten-target seek spread (forward and backward jumps across the clip)
twice in one group — once on the binary-search-backed
`seek_to_via_index` path and once on the linear disk-walking fallback —
quantifying the chunk-index payoff (the local run measured roughly a
7× speedup: about 7.0 µs for the ten indexed seeks versus 49.4 µs for
the same spread walked linearly). `mux_write` benches the full
`write_header` + 1116-pair `write_packet` + `write_trailer` write path.
A shared `benches/common/mod.rs` carries the muxer driver (an
`Arc<Mutex<Cursor<Vec<u8>>>>`-backed `WriteSeek` so the bytes survive
the muxer's `Box<dyn WriteSeek>` type erasure) and the `comedian.amv`
`StreamInfo` pair. Run with
`cargo bench -p oxideav-amv --bench <name>`.

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
(`AmvHeader`, `AmvDuration`, `AmvWaveFormat`, `AmvAudioPreamble`,
`AmvVideoFrame`, `ChunkHeader`, `ChunkKind`, plus the chunk-tag /
trailer / JPEG-marker constants and the `validate_video_payload_shape` /
`validate_video_payload_no_internal_markers` /
`validate_movi_interleave` free functions) for tooling that wants to
inspect AMV files without the framework dependency at the demuxer
level.

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
