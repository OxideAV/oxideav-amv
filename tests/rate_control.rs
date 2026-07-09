//! Measured bitrate-vs-target validation of the §4a rate-controlled
//! encode on real device content.
//!
//! The workspace stages a real device-origin `comedian.amv`
//! (128×96 @ 12 fps, 1116 frames — trace §2). These tests decode its
//! frames with the in-crate decoder, re-encode them through
//! [`oxideav_amv::encode_frame_yuv420p_with_budget`] under an
//! [`oxideav_amv::AmvRateController`], and **measure** the delivered
//! payload bitrate against the requested target:
//!
//! * at a *binding* target (60 % of the device stream's own payload
//!   rate) the delivered average must hold the target from below while
//!   using most of it, every frame must fit its budget, and the
//!   trimmed frames must still decode close to the source decode;
//! * at a *generous* target (2× the device rate) rate control must be
//!   a byte-level no-op: every payload equals the unconstrained
//!   re-encode.
//!
//! Both tests skip when the fixture is not staged (e.g. on per-crate
//! CI, which checks out this repository alone). No external binary is
//! used — decodability is judged by the crate's own §4a decoder, whose
//! fidelity is separately pinned against a black-box reference decoder
//! in `tests/decode_to_pixels.rs`.

use std::io::{Cursor, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use oxideav_amv::{
    decode_frame_from_payload, decode_frame_yuv420p_from_payload, encode_audio_payload,
    encode_frame_yuv420p, encode_frame_yuv420p_with_budget, AmvDemuxer, AmvRateController,
};
use oxideav_core::{Demuxer, Error};

fn comedian_fixture() -> Option<PathBuf> {
    let crate_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/comedian.amv");
    if crate_path.exists() {
        return Some(crate_path);
    }
    let workspace_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/container/amv/fixtures/comedian.amv");
    if workspace_path.exists() {
        return Some(workspace_path);
    }
    None
}

/// Pull the first `max_frames` video payloads (bare `00dc` bodies) out
/// of the staged fixture, plus the parsed header.
fn fixture_video_payloads(max_frames: usize) -> Option<(oxideav_amv::AmvHeader, Vec<Vec<u8>>)> {
    let path = comedian_fixture()?;
    let bytes = std::fs::read(path).expect("read comedian fixture");
    let mut d = AmvDemuxer::open(std::io::Cursor::new(bytes)).expect("open fixture");
    let header = *d.header();
    let mut payloads = Vec::new();
    loop {
        match d.next_packet() {
            Ok(p) if p.stream_index == 0 => {
                payloads.push(p.data.clone());
                if payloads.len() >= max_frames {
                    break;
                }
            }
            Ok(_) => {}
            Err(Error::Eof) => break,
            Err(e) => panic!("fixture walk error: {e:?}"),
        }
    }
    Some((header, payloads))
}

#[test]
fn binding_target_holds_measured_bitrate_on_comedian() {
    let Some((header, payloads)) = fixture_video_payloads(300) else {
        eprintln!("skipping AMV rate-control fixture test: comedian.amv not staged");
        return;
    };
    let fps = header.fps;
    assert_eq!(fps, 12, "trace §2: comedian is a 12 fps profile");

    // The device stream's own payload rate over this window.
    let device_total: usize = payloads.iter().map(Vec::len).sum();
    let device_avg = device_total / payloads.len();

    // Ask for 60 % of it — a genuinely binding target.
    let target_per_frame = device_avg * 60 / 100;
    let target_bps = (target_per_frame * 8 * fps as usize) as u64;
    let mut rc = AmvRateController::from_video_bitrate(target_bps, fps).expect("controller");
    assert_eq!(rc.target_bytes_per_frame(), target_per_frame as u64);

    let mut worst_mae = 0f64;
    for (i, payload) in payloads.iter().enumerate() {
        let yuv = decode_frame_yuv420p_from_payload(&header, payload).expect("decode source");
        let budget = rc.frame_budget();
        let b = encode_frame_yuv420p_with_budget(
            header.width,
            header.height,
            &yuv.y,
            &yuv.cb,
            &yuv.cr,
            budget,
        )
        .expect("budgeted encode");
        assert!(
            b.within_budget,
            "frame {i}: a 60% budget ({budget} B) must be achievable on real content"
        );
        assert!(b.payload.len() <= budget, "frame {i} exceeds its budget");
        rc.note_frame(b.payload.len());

        // Spot-check fidelity every 50th frame: the trimmed re-encode
        // must stay close to the source decode.
        if i % 50 == 0 {
            let src = decode_frame_from_payload(&header, payload).expect("src decode");
            let out = decode_frame_from_payload(&header, &b.payload).expect("trimmed decode");
            let mae = src
                .rgb
                .iter()
                .zip(&out.rgb)
                .map(|(&a, &b)| (a as f64 - b as f64).abs())
                .sum::<f64>()
                / src.rgb.len() as f64;
            worst_mae = worst_mae.max(mae);
        }
    }

    let avg = rc.average_bytes_per_frame();
    eprintln!(
        "comedian 300-frame window: device {device_avg} B/frame, target {target_per_frame} \
         B/frame ({target_bps} bps), delivered {avg:.1} B/frame ({:.0} bps), worst sampled \
         MAE {worst_mae:.2}/channel",
        rc.achieved_bits_per_sec(fps)
    );
    assert!(
        avg <= target_per_frame as f64,
        "measured average {avg:.1} B/frame must hold the {target_per_frame} B/frame target"
    );
    assert!(
        avg >= target_per_frame as f64 * 0.85,
        "measured average {avg:.1} B/frame should use most of the {target_per_frame} B/frame target"
    );
    let achieved = rc.achieved_bits_per_sec(fps);
    assert!(
        achieved <= target_bps as f64 && achieved >= target_bps as f64 * 0.85,
        "achieved {achieved:.0} bps vs target {target_bps} bps out of tolerance"
    );
    assert!(
        worst_mae < 25.0,
        "worst sampled MAE {worst_mae:.2}/channel too high at a 60% rate target"
    );
}

#[test]
fn generous_target_is_a_byte_level_no_op_on_comedian() {
    let Some((header, payloads)) = fixture_video_payloads(50) else {
        eprintln!("skipping AMV rate-control fixture test: comedian.amv not staged");
        return;
    };
    let fps = header.fps;
    let device_total: usize = payloads.iter().map(Vec::len).sum();
    let device_avg = device_total / payloads.len();

    // 2× the device rate: never binds (our unconstrained re-encode of
    // decoded device frames tracks the device's own frame sizes).
    let target_bps = (device_avg * 2 * 8 * fps as usize) as u64;
    let mut rc = AmvRateController::from_video_bitrate(target_bps, fps).expect("controller");

    for (i, payload) in payloads.iter().enumerate() {
        let yuv = decode_frame_yuv420p_from_payload(&header, payload).expect("decode source");
        let unconstrained =
            encode_frame_yuv420p(header.width, header.height, &yuv.y, &yuv.cb, &yuv.cr)
                .expect("unconstrained encode");
        let b = encode_frame_yuv420p_with_budget(
            header.width,
            header.height,
            &yuv.y,
            &yuv.cb,
            &yuv.cr,
            rc.frame_budget(),
        )
        .expect("budgeted encode");
        assert!(b.within_budget, "frame {i}: generous budget must fit");
        assert_eq!(
            b.payload, unconstrained,
            "frame {i}: an unbinding budget must return the unconstrained bytes"
        );
        rc.note_frame(b.payload.len());
    }
    // And the booked rate stays at the natural rate, far below target.
    assert!(rc.achieved_bits_per_sec(fps) < target_bps as f64 * 0.75);
}

// ─────────────────────────────────────────────────────────────────────
// End-to-end registry pipeline: CodecParameters::bit_rate → registry
// encoder → AmvMuxer → demux. Synthetic content, so this also runs on
// per-crate CI (no fixture needed).
// ─────────────────────────────────────────────────────────────────────

/// A `WriteSeek` whose bytes stay reachable after the boxed writer the
/// muxer owns is dropped.
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
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.0.lock().unwrap().seek(pos)
    }
}

/// Deterministic noisy YUV420P planes (per-frame seed) with enough AC
/// energy that a byte budget binds.
fn noisy_yuv(w: u32, h: u32, seed: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let cw = w.div_ceil(2) as usize;
    let ch = h.div_ceil(2) as usize;
    let mut lcg = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut next = || {
        lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (lcg >> 24) as u8
    };
    let mut y = vec![0u8; (w * h) as usize];
    for (i, s) in y.iter_mut().enumerate() {
        let base = ((i as u32 % w) * 2) as i32 % 160 + 48;
        *s = (base + (next() as i32 - 128) / 3).clamp(0, 255) as u8;
    }
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for s in cb.iter_mut() {
        *s = 100 + next() % 56;
    }
    for s in cr.iter_mut() {
        *s = 100 + next() % 56;
    }
    (y, cb, cr)
}

#[test]
fn registry_pipeline_writes_a_rate_controlled_amv_file() {
    use oxideav_core::{
        CodecId, CodecParameters, Frame, MediaType, PixelFormat, Rational, RuntimeContext,
        StreamInfo, TimeBase, VideoFrame, VideoPlane, WriteSeek,
    };

    let (w, h, fps) = (64u32, 48u32, 12u32);
    let n_frames = 36u32; // 3 s at 12 fps
    let samples_per_block = 22_050 / fps; // §4b: one frame-interval per block

    let mut ctx = RuntimeContext::new();
    oxideav_amv::register(&mut ctx);

    // Measure the natural rate of the synthetic content through the
    // registry encoder, then request roughly half of it.
    let mut probe_params = CodecParameters::video(CodecId::new("amv_video"));
    probe_params.width = Some(w);
    probe_params.height = Some(h);
    probe_params.pixel_format = Some(PixelFormat::Yuv420P);
    probe_params.frame_rate = Some(Rational::new(fps as i64, 1));
    let mut probe_enc = ctx.codecs.first_encoder(&probe_params).expect("encoder");
    let mut natural_total = 0usize;
    for i in 0..n_frames {
        let (y, cb, cr) = noisy_yuv(w, h, i);
        let cw = w.div_ceil(2) as usize;
        probe_enc
            .send_frame(&Frame::Video(VideoFrame {
                pts: Some(i as i64),
                planes: vec![
                    VideoPlane {
                        stride: w as usize,
                        data: y,
                    },
                    VideoPlane {
                        stride: cw,
                        data: cb,
                    },
                    VideoPlane {
                        stride: cw,
                        data: cr,
                    },
                ],
            }))
            .unwrap();
        natural_total += probe_enc.receive_packet().unwrap().data.len();
    }
    let target_per_frame = natural_total / n_frames as usize / 2;
    let bit_rate = (target_per_frame * 8 * fps as usize) as u64;

    // The real rate-controlled encoder.
    let mut vparams = probe_params.clone();
    vparams.bit_rate = Some(bit_rate);
    let mut enc = ctx
        .codecs
        .first_encoder(&vparams)
        .expect("rate-controlled encoder resolves through the registry");

    // Streams for the muxer (video then audio).
    let mut aparams = CodecParameters::audio(CodecId::new("adpcm_amv"));
    aparams.media_type = MediaType::Audio;
    aparams.sample_rate = Some(22_050);
    aparams.channels = Some(1);
    let streams = vec![
        StreamInfo {
            index: 0,
            time_base: TimeBase::new(1, fps as i64),
            duration: None,
            start_time: Some(0),
            params: enc.output_params().clone(),
        },
        StreamInfo {
            index: 1,
            time_base: TimeBase::new(1, 22_050),
            duration: None,
            start_time: Some(0),
            params: aparams,
        },
    ];

    let shared = SharedBuf(Arc::new(Mutex::new(Cursor::new(Vec::new()))));
    let writer: Box<dyn WriteSeek> = Box::new(shared.clone());
    let mut mux = ctx
        .containers
        .open_muxer("amv", writer, &streams)
        .expect("registry muxer opens");
    mux.write_header().unwrap();

    let mut video_total = 0usize;
    for i in 0..n_frames {
        let (y, cb, cr) = noisy_yuv(w, h, i);
        let cw = w.div_ceil(2) as usize;
        enc.send_frame(&Frame::Video(VideoFrame {
            pts: Some(i as i64),
            planes: vec![
                VideoPlane {
                    stride: w as usize,
                    data: y,
                },
                VideoPlane {
                    stride: cw,
                    data: cb,
                },
                VideoPlane {
                    stride: cw,
                    data: cr,
                },
            ],
        }))
        .unwrap();
        let pkt = enc.receive_packet().unwrap();
        video_total += pkt.data.len();
        mux.write_packet(&pkt).unwrap();

        // One §4b audio block per video frame: a quiet synthetic tone.
        let samples: Vec<i16> = (0..samples_per_block)
            .map(|k| ((k as f32 * 0.11).sin() * 900.0) as i16)
            .collect();
        let audio_payload = encode_audio_payload(&samples);
        let mut apkt = oxideav_core::Packet::new(1, TimeBase::new(1, 22_050), audio_payload);
        apkt.pts = Some((i * samples_per_block) as i64);
        mux.write_packet(&apkt).unwrap();
    }
    mux.write_trailer().unwrap();
    drop(mux);
    let bytes = shared.0.lock().unwrap().get_ref().clone();

    // The written file is a strict-valid AMV at the requested rate.
    let mut d = AmvDemuxer::open_strict(Cursor::new(bytes)).expect("strict open");
    assert_eq!(d.header().width, w);
    assert_eq!(d.header().height, h);
    assert_eq!(d.header().fps, fps);
    let mut n_video = 0u32;
    let mut n_audio = 0u32;
    let mut demuxed_video_total = 0usize;
    let mut vdec = ctx
        .codecs
        .first_decoder(&probe_params)
        .expect("registry video decoder");
    loop {
        match d.next_packet() {
            Ok(p) if p.stream_index == 0 => {
                demuxed_video_total += p.data.len();
                // Every rate-controlled frame decodes through the
                // registry decoder.
                vdec.send_packet(&p).unwrap();
                let Frame::Video(f) = vdec.receive_frame().unwrap() else {
                    panic!("expected video frame");
                };
                assert_eq!(f.planes[0].data.len(), (w * h) as usize);
                n_video += 1;
            }
            Ok(p) if p.stream_index == 1 => {
                n_audio += 1;
                assert_eq!(p.data.len(), 8 + (samples_per_block as usize).div_ceil(2));
            }
            Ok(_) => panic!("unexpected stream"),
            Err(Error::Eof) => break,
            Err(e) => panic!("demux error: {e:?}"),
        }
    }
    assert_eq!((n_video, n_audio), (n_frames, n_frames));
    assert!(d.movi_interleave_balanced());
    // §2 duration: 36 frames ÷ 12 fps = 3 s.
    assert_eq!(d.header().duration().total_seconds(), 3);
    assert_eq!(demuxed_video_total, video_total);
    // Measured file-level video rate holds the target.
    let target_total = target_per_frame * n_frames as usize;
    assert!(
        video_total <= target_total,
        "video payload {video_total} must hold the {target_total}-byte stream target"
    );
    assert!(
        video_total * 4 >= target_total * 3,
        "video payload {video_total} should use most of the {target_total}-byte stream target"
    );
    assert!(
        video_total * 10 < natural_total * 7,
        "rate control must actually bind ({video_total} vs natural {natural_total})"
    );
}
