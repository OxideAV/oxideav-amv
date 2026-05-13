//! Seek tests for the AMV demuxer.
//!
//! AMV has no built-in `idx1`-style chunk index, so the demuxer builds
//! a lazy table of every `00dc` chunk's byte offset + per-stream PTS
//! counters on first seek. The tests below exercise the three points
//! that matter:
//!
//! * `seek_to_zero_resets_to_start` — `pts <= 0` lands on the very
//!   first video chunk; the cursor is back at `movi_start` and the
//!   next `00dc` packet has `pts == 0`.
//! * `seek_lands_at_video_frame_boundary` — every video frame is a
//!   keyframe (AMV video is intra-only), so a seek to an interior pts
//!   must land on the exact `00dc` whose v_pts equals the target.
//! * `seek_past_end_clamps` — overshooting the last frame clamps to
//!   the final video entry instead of erroring.
//!
//! The tests use two fixtures: a deterministic hand-crafted blob
//! (three V/A pairs, no ffmpeg required) for the structural assertions
//! and an ffmpeg-generated 30-frame AMV file when `/usr/bin/ffmpeg` or
//! `/opt/homebrew/bin/ffmpeg` is available. A 30-frame fixture is
//! checked into `tests/fixtures/` so CI doesn't need ffmpeg.

use std::io::Cursor;
use std::path::Path;

use oxideav_core::{ContainerRegistry, Error, MediaType, NullCodecResolver};

// ---------- hand-crafted multi-frame AMV ------------------------------------

fn put_u32_le(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u16_le(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// Build a deterministic AMV byte stream with `n_frames` interleaved
/// V/A pairs. Each video chunk is 4 bytes (SOI+EOI marker pair) and
/// each audio chunk is the minimum 12 bytes (8-byte header + 4 bytes
/// of nibbles → 8 samples). Frame `i` carries its index in the first
/// payload byte so we can verify a seek landed on the right frame
/// without needing to decode JPEG.
fn make_multi_frame_amv(n_frames: usize) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();

    // RIFF <0> AMV
    buf.extend_from_slice(b"RIFF");
    put_u32_le(&mut buf, 0);
    buf.extend_from_slice(b"AMV ");

    // LIST <0> hdrl
    buf.extend_from_slice(b"LIST");
    put_u32_le(&mut buf, 0);
    buf.extend_from_slice(b"hdrl");

    // amvh <56> body — only width@32, height@36, fps_den@40, fps_num@44 matter.
    buf.extend_from_slice(b"amvh");
    put_u32_le(&mut buf, 56);
    let mut amvh = vec![0u8; 56];
    amvh[32..36].copy_from_slice(&16u32.to_le_bytes());
    amvh[36..40].copy_from_slice(&16u32.to_le_bytes());
    amvh[40..44].copy_from_slice(&1u32.to_le_bytes()); // fps_den
    amvh[44..48].copy_from_slice(&15u32.to_le_bytes()); // fps_num
    buf.extend_from_slice(&amvh);

    // LIST <0> strl (video strh+strf)
    buf.extend_from_slice(b"LIST");
    put_u32_le(&mut buf, 0);
    buf.extend_from_slice(b"strl");
    buf.extend_from_slice(b"strh");
    put_u32_le(&mut buf, 56);
    buf.extend_from_slice(&[0u8; 56]);
    buf.extend_from_slice(b"strf");
    put_u32_le(&mut buf, 36);
    buf.extend_from_slice(&[0u8; 36]);

    // LIST <0> strl (audio strh+strf=WAVEFORMATEX 20)
    buf.extend_from_slice(b"LIST");
    put_u32_le(&mut buf, 0);
    buf.extend_from_slice(b"strl");
    buf.extend_from_slice(b"strh");
    put_u32_le(&mut buf, 48);
    buf.extend_from_slice(&[0u8; 48]);
    buf.extend_from_slice(b"strf");
    put_u32_le(&mut buf, 20);
    let mut wfx = vec![0u8; 20];
    wfx[0..2].copy_from_slice(&1u16.to_le_bytes());
    wfx[2..4].copy_from_slice(&1u16.to_le_bytes());
    wfx[4..8].copy_from_slice(&22_050u32.to_le_bytes());
    wfx[8..12].copy_from_slice(&22_050u32.to_le_bytes());
    put_u16_le(&mut wfx[12..14].to_vec(), 1);
    wfx[12..14].copy_from_slice(&1u16.to_le_bytes());
    wfx[14..16].copy_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(&wfx);

    // LIST <0> movi
    buf.extend_from_slice(b"LIST");
    put_u32_le(&mut buf, 0);
    buf.extend_from_slice(b"movi");

    for i in 0..n_frames {
        // 00dc <4> [i, FF, D8, D9]  ← marker byte + SOI hi + EOI lo/hi
        // (the decoder isn't run; we only need the tag walker to see
        // a valid-shaped 00dc chunk and remember its position.)
        buf.extend_from_slice(b"00dc");
        put_u32_le(&mut buf, 4);
        buf.extend_from_slice(&[i as u8, 0xD8, 0xFF, 0xD9]);

        // 01wb <12>  (8 header + 4 data nibble bytes = 8 samples)
        buf.extend_from_slice(b"01wb");
        put_u32_le(&mut buf, 12);
        let mut audio = vec![0u8; 8];
        audio[0] = i as u8; // predictor lo — also marks which frame this came from
        audio.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(&audio);
    }

    // Trailer.
    buf.extend_from_slice(b"AMV_");
    buf.extend_from_slice(b"END_");
    buf
}

fn open_demuxer(blob: Vec<u8>) -> Box<dyn oxideav_core::Demuxer> {
    let mut reg = ContainerRegistry::new();
    oxideav_amv::register_containers(&mut reg);
    reg.open_demuxer("amv", Box::new(Cursor::new(blob)), &NullCodecResolver)
        .expect("open AMV demuxer")
}

// ---------- always-on tests against the hand-crafted blob -------------------

#[test]
fn seek_to_zero_resets_to_start() {
    let blob = make_multi_frame_amv(5);
    let mut demux = open_demuxer(blob);

    // Drain the first two video frames so the demuxer's internal
    // cursor is well past the start.
    let v0 = demux.next_packet().expect("V0 pre-seek");
    assert_eq!(v0.stream_index, 0);
    assert_eq!(v0.pts, Some(0));
    let _a0 = demux.next_packet().expect("A0 pre-seek");
    let v1 = demux.next_packet().expect("V1 pre-seek");
    assert_eq!(v1.pts, Some(1));

    // Seek back to pts=0 on the video stream.
    let landed = demux.seek_to(0, 0).expect("seek_to(0, 0)");
    assert_eq!(landed, 0, "seek_to(0,0) must land on v_pts=0");

    // Next packet must be the first video chunk, identifiable by its
    // marker byte (frame_index == 0).
    let pkt = demux.next_packet().expect("post-seek packet");
    assert_eq!(pkt.stream_index, 0);
    assert_eq!(pkt.pts, Some(0));
    assert_eq!(pkt.data[0], 0, "post-seek video payload must be frame 0");

    // And the next audio packet must reset to a_pts=0.
    let apkt = demux.next_packet().expect("post-seek audio");
    assert_eq!(apkt.stream_index, 1);
    assert_eq!(apkt.pts, Some(0));
}

#[test]
fn seek_lands_at_video_frame_boundary() {
    let blob = make_multi_frame_amv(8);
    let mut demux = open_demuxer(blob);

    // Cold-seek (no prior reads) to v_pts=3.
    let landed = demux.seek_to(0, 3).expect("seek_to(0, 3)");
    assert_eq!(
        landed, 3,
        "AMV is intra-only — every frame is a keyframe; landed pts must equal target"
    );
    let pkt = demux.next_packet().expect("packet after seek");
    assert_eq!(pkt.stream_index, 0);
    assert_eq!(pkt.pts, Some(3));
    assert_eq!(pkt.data[0], 3, "must be the marker byte of frame 3");

    // Audio-stream seek: in our blob each audio chunk advances a_pts
    // by 8 samples (4 nibble bytes × 2). Pre-V_i audio cumulative is
    // 8 * i. Asking for pts=17 should land on V_2 (a_pts=16).
    let landed_a = demux.seek_to(1, 17).expect("seek_to(1, 17)");
    assert_eq!(landed_a, 16, "audio seek should round down to V_2's a_pts");
    let apkt = demux.next_packet().expect("V chunk after audio seek");
    // The demuxer always lands on a video chunk; the next packet is
    // therefore the matching 00dc, not the audio chunk.
    assert_eq!(apkt.stream_index, 0);
    assert_eq!(apkt.pts, Some(2));
    assert_eq!(apkt.data[0], 2);

    // And then the paired audio chunk.
    let aphys = demux.next_packet().expect("audio after V_2");
    assert_eq!(aphys.stream_index, 1);
    assert_eq!(aphys.pts, Some(16));
}

#[test]
fn seek_past_end_clamps() {
    let n = 6usize;
    let blob = make_multi_frame_amv(n);
    let mut demux = open_demuxer(blob);

    // Way past the last frame index.
    let landed = demux
        .seek_to(0, 9999)
        .expect("seek_to(0, 9999) must clamp, not error");
    assert_eq!(
        landed,
        (n - 1) as i64,
        "must clamp to the last video frame's pts"
    );
    let pkt = demux.next_packet().expect("packet after clamp seek");
    assert_eq!(pkt.stream_index, 0);
    assert_eq!(pkt.pts, Some((n - 1) as i64));
    assert_eq!(pkt.data[0], (n - 1) as u8);

    // Audio overshoot also clamps to last indexed entry's a_pts.
    let landed_a = demux
        .seek_to(1, i64::MAX / 2)
        .expect("audio overshoot must clamp");
    assert_eq!(landed_a, (8 * (n as i64 - 1)));
}

#[test]
fn seek_invalid_stream_index_errors() {
    let blob = make_multi_frame_amv(3);
    let mut demux = open_demuxer(blob);
    match demux.seek_to(99, 0) {
        Err(Error::InvalidData(_)) => {}
        other => panic!(
            "expected Error::InvalidData for out-of-range stream, got {:?}",
            other
        ),
    }
}

// ---------- ffmpeg-generated fixture ---------------------------------------

const FIXTURE_REL: &str = "tests/fixtures/testsrc_30frames.amv";

#[test]
fn seek_on_ffmpeg_fixture_lands_at_expected_frame() {
    // CARGO_MANIFEST_DIR is set during `cargo test`; outside cargo
    // (e.g. doc-tests) we just skip.
    let manifest = match std::env::var("CARGO_MANIFEST_DIR") {
        Ok(s) => s,
        Err(_) => return,
    };
    let path = Path::new(&manifest).join(FIXTURE_REL);
    if !path.exists() {
        // Fixture isn't checked in for some reason; not a hard fail —
        // the deterministic in-memory blob already covers the seek
        // semantics. Print a hint and move on.
        eprintln!(
            "AMV ffmpeg fixture missing at {} — skipping. Regenerate with:\n  \
             ffmpeg -f lavfi -i testsrc=size=64x64:rate=15:duration=2 \\\n  \
                    -f lavfi -i anullsrc=r=22050:cl=mono -shortest \\\n  \
                    -c:v amv -c:a adpcm_ima_amv -ar 22050 -block_size 1470 {}",
            path.display(),
            path.display(),
        );
        return;
    }

    let bytes = std::fs::read(&path).expect("read fixture");
    let mut demux = open_demuxer(bytes);

    // Sanity: stream layout matches the AMV invariant.
    let streams = demux.streams().to_vec();
    assert_eq!(streams.len(), 2);
    assert_eq!(streams[0].params.media_type, MediaType::Video);
    assert_eq!(streams[1].params.media_type, MediaType::Audio);

    // Seek to v_pts=10 in a 30-frame fixture, expect to land exactly
    // on frame 10 (intra-only ⇒ every frame is a keyframe).
    let landed = demux.seek_to(0, 10).expect("seek to v_pts=10");
    assert_eq!(landed, 10);
    let pkt = demux.next_packet().expect("packet after fixture seek");
    assert_eq!(pkt.stream_index, 0);
    assert_eq!(pkt.pts, Some(10));

    // Round-trip: re-seek to 0 and confirm we get pts=0 back.
    let landed0 = demux.seek_to(0, 0).expect("re-seek to 0");
    assert_eq!(landed0, 0);
    let pkt0 = demux.next_packet().expect("first frame after re-seek");
    assert_eq!(pkt0.pts, Some(0));
}
