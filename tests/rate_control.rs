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

use std::path::{Path, PathBuf};

use oxideav_amv::{
    decode_frame_from_payload, decode_frame_yuv420p_from_payload, encode_frame_yuv420p,
    encode_frame_yuv420p_with_budget, AmvDemuxer, AmvRateController,
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
