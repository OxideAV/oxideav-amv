//! Shared bench helpers for the oxideav-amv depth-mode (bench) round.
//!
//! Every bench input is synthesised in-bench through the crate's own
//! **public** `AmvMuxer` write path — no committed fixture file and no
//! crate-private helper. The muxer emits a byte-faithful AMV stream
//! (trace §1..§4c), which the demux / index / seek benches then walk.
//!
//! The synthetic payloads mirror the trace's worked-example shapes:
//! each `00dc` body is a minimal SOI..EOI JPEG bracket and each `01wb`
//! body is the 8-byte §4b preamble (`state = 0`,
//! `decoded_sample_count = 1837`) followed by a few nibble-coded bytes.
//! Sizes are deliberately small so the benches measure container-walk
//! overhead (chunk-header parse, no-padding advance, PTS accounting)
//! rather than payload memcpy.

use std::io::{Cursor, Seek, Write};
use std::sync::{Arc, Mutex};

use oxideav_core::{
    CodecId, CodecParameters, MediaType, Muxer, Packet, PixelFormat, Rational, StreamInfo,
    TimeBase, WriteSeek,
};

use oxideav_amv::AmvMuxer;

/// A `WriteSeek` that keeps the underlying `Vec<u8>` reachable after the
/// `Box<dyn WriteSeek>` the muxer owns is dropped. `AmvMuxer::open`
/// erases the writer's concrete type, so a plain `Cursor<Vec<u8>>` could
/// never hand its bytes back; the `Arc<Mutex<…>>` clone we retain does.
#[derive(Clone)]
struct SharedBuf(Arc<Mutex<Cursor<Vec<u8>>>>);

impl Write for SharedBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.lock().unwrap().flush()
    }
}

impl Seek for SharedBuf {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.0.lock().unwrap().seek(pos)
    }
}

/// Build the `comedian.amv` `[video, audio]` `StreamInfo` pair the muxer
/// needs (128×96 @ 12 fps video, 22 050 Hz mono audio).
pub fn comedian_streams() -> Vec<StreamInfo> {
    let mut video_params = CodecParameters::video(CodecId::new("mjpeg"));
    video_params.media_type = MediaType::Video;
    video_params.width = Some(128);
    video_params.height = Some(96);
    video_params.pixel_format = Some(PixelFormat::Yuv420P);
    video_params.frame_rate = Some(Rational::new(12, 1));
    let video = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 12),
        duration: None,
        start_time: Some(0),
        params: video_params,
    };

    let mut audio_params = CodecParameters::audio(CodecId::new("adpcm_amv"));
    audio_params.media_type = MediaType::Audio;
    audio_params.sample_rate = Some(22_050);
    audio_params.channels = Some(1);
    let audio = StreamInfo {
        index: 1,
        time_base: TimeBase::new(1, 22_050),
        duration: None,
        start_time: Some(0),
        params: audio_params,
    };

    vec![video, audio]
}

/// One synthetic `00dc` video body: a self-contained SOI..EOI JPEG
/// bracket with a 4-byte index marker so the trace §4a payload-shape
/// invariant (`FF D8` … `FF D9`) holds.
fn video_body(i: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(8);
    v.extend_from_slice(&[0xFF, 0xD8]); // SOI
    v.extend_from_slice(&i.to_le_bytes()); // entropy-window stand-in
    v.extend_from_slice(&[0xFF, 0xD9]); // EOI
    v
}

/// One synthetic `01wb` audio body: the trace §4b 8-byte preamble
/// (`state = 0`, `decoded_sample_count = 1837` = 22 050 ÷ 12) plus a
/// few nibble-coded bytes.
fn audio_body() -> Vec<u8> {
    let mut a = Vec::with_capacity(12);
    a.extend_from_slice(&0u32.to_le_bytes()); // §4b per-block state
    a.extend_from_slice(&1837u32.to_le_bytes()); // §4b decoded sample count
    a.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
    a
}

/// Mux a full AMV file (§1 prelude, `n_pairs` video-first `00dc`/`01wb`
/// pairs, §4c `AMV_END_` trailer) through the public `AmvMuxer` and
/// return the recovered bytes. Used as the shared input for the
/// demux / index / seek benches.
pub fn build_amv_bytes(n_pairs: usize) -> Vec<u8> {
    let streams = comedian_streams();
    let shared = SharedBuf(Arc::new(Mutex::new(Cursor::new(Vec::<u8>::new()))));
    let writer: Box<dyn WriteSeek> = Box::new(shared.clone());
    let mut mux = AmvMuxer::open(writer, &streams).expect("amv muxer open");

    mux.write_header().expect("write_header");
    for i in 0..n_pairs {
        mux.write_packet(&Packet::new(0, TimeBase::new(1, 12), video_body(i as u32)))
            .expect("write video packet");
        mux.write_packet(&Packet::new(1, TimeBase::new(1, 22_050), audio_body()))
            .expect("write audio packet");
    }
    mux.write_trailer().expect("write_trailer");

    let guard = shared.0.lock().unwrap();
    guard.get_ref().clone()
}
