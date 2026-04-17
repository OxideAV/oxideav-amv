# oxideav-amv

Pure-Rust **AMV** codec + container — the weird, deliberately-broken
RIFF/AVI variant used by cheap Chinese MP3/MP4 players. Ships a video
decoder + encoder (stripped-header Motion-JPEG at Annex-K Q50), an
IMA-ADPCM audio decoder + encoder (the AMV variant, mono), and a
container demuxer that tolerates the zeroed top-level size fields
real AMV files carry. Zero C dependencies.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.0"
oxideav-codec = "0.0"
oxideav-container = "0.0"
oxideav-amv = "0.0"
```

## Quick use

AMV carries one video and one audio stream, strictly interleaved.
Open the file as a container, pull packets, dispatch to the matching
decoder:

```rust
use oxideav_codec::CodecRegistry;
use oxideav_container::ContainerRegistry;
use oxideav_core::{Frame, MediaType};

let mut codecs = CodecRegistry::new();
let mut containers = ContainerRegistry::new();
oxideav_amv::register(&mut codecs, &mut containers);

let input: Box<dyn oxideav_container::ReadSeek> = Box::new(
    std::io::Cursor::new(std::fs::read("clip.amv")?),
);
let mut dmx = containers.open("amv", input)?;
let streams = dmx.streams().to_vec();
let mut vdec = codecs.make_decoder(&streams[0].params)?;
let mut adec = codecs.make_decoder(&streams[1].params)?;

loop {
    match dmx.next_packet() {
        Ok(pkt) => {
            let dec = if pkt.stream_index == 0 { &mut vdec } else { &mut adec };
            dec.send_packet(&pkt)?;
            match dec.receive_frame()? {
                Frame::Video(vf) => { /* Yuv420P, already flipped right-side up */ }
                Frame::Audio(af) => { /* S16, mono, packed LE */ }
            }
        }
        Err(oxideav_core::Error::Eof) => break,
        Err(e) => return Err(e.into()),
    }
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Encoders

Both codecs encode too — useful for building a real AMV file or
round-trip testing:

```rust
// Video: Yuv420P in, AMV 00dc-shaped payload out (FF D8 + entropy + FF D9).
let mut v_params = CodecParameters::video(CodecId::new("amv"));
v_params.width = Some(w);
v_params.height = Some(h);
v_params.pixel_format = Some(PixelFormat::Yuv420P);
let mut venc = codecs.make_encoder(&v_params)?;
venc.send_frame(&Frame::Video(frame))?;
let v_pkt = venc.receive_packet()?;

// Audio: S16 mono in, AMV 01wb-shaped payload out (8-byte header + nibbles).
let mut a_params = CodecParameters::audio(CodecId::new("adpcm_ima_amv"));
a_params.sample_rate = Some(22_050);
a_params.channels = Some(1);
a_params.sample_format = Some(SampleFormat::S16);
let mut aenc = codecs.make_encoder(&a_params)?;
aenc.send_frame(&Frame::Audio(audio))?;
let a_pkt = aenc.receive_packet()?;
```

## Codec / container IDs

- Video codec: `"amv"`. Accepted pixel format `Yuv420P`. Intra-only,
  lossy. Internally a stripped-header baseline JPEG at Q50 (the fixed
  Annex-K tables the AMV decoder hard-codes) with the frame stored
  upside-down on disk — the decoder flips it back, the encoder flips
  before encode. Non-4:2:0 input is rejected; Q is not a tunable
  (Q50 is a protocol-level constraint, since the decoder reconstructs
  the stripped DQT from the standard Annex-K table).
- Audio codec: `"adpcm_ima_amv"`. Mono S16 at up to 48 kHz. Each
  packet carries its own 8-byte IMA-ADPCM header (initial predictor,
  initial step-index, 5 reserved bytes) so every chunk is
  self-contained on decode. Multi-channel input is rejected.
- Container: `"amv"`, matches `.amv` by extension and the
  `RIFF....AMV ` magic. Tolerant to the zero-size top-level
  LIST/RIFF chunks that real AMV files ship with; walks by tag.
  Stream 0 is always video, stream 1 is always audio (AMV enforces
  strict V-A interleaving).

## Limitations

- No muxer yet — this crate reads AMV containers and encodes the raw
  codec payloads, but does not write the `RIFF....AMV ` envelope.
- Video quality is fixed at Q50. Other Q factors would produce scaled
  quant tables that the decoder's hand-rolled header cannot recover.
- Audio is mono-only at up to 48 kHz, matching the AMV variant.

## License

MIT — see [LICENSE](LICENSE).
