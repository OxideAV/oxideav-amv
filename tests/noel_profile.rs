//! Cross-profile container conformance against the second staged
//! fixture, `noel-son-lumiere.amv` — the 96 × 64 @ 16 fps device profile
//! (trace "Samples parsed" table: 2928 `00dc` / 2928 `01wb` chunks,
//! packed duration 3:02).
//!
//! `comedian.amv` established every §1–§4c container rule; this suite
//! proves those rules are properties of the *format*, not of one file:
//! the strict sentinel open, the §4 1:1 video-first interleave at 2928
//! pairs, the no-padding chunk walk terminating at `AMV_END_`, the §4b
//! preamble field widths (the `+0x03 = 0xAA` per-file constant that
//! settles the one-byte step-index reading), and the §4b closing
//! observation that this profile's `amvh` duration is written truncated
//! by one second (3:02 against a derived 2928 ÷ 16 = 183 s = 3:03).
//!
//! Skipped when the fixture is not staged (e.g. in CI, which does not
//! carry the workspace docs tree).

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use oxideav_amv::{
    validate_video_payload_shape, AmvAudioPreamble, AmvDemuxer, DurationConsistency, MoviPayload,
    MoviPayloadIter, AMV_END_TRAILER, IMA_STEP_INDEX_MAX, JPEG_SOI,
};
use oxideav_core::{Demuxer, Error};

fn noel_fixture() -> Option<PathBuf> {
    let crate_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/noel-son-lumiere.amv");
    if crate_path.exists() {
        return Some(crate_path);
    }
    let workspace_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/container/amv/fixtures/noel-son-lumiere.amv");
    if workspace_path.exists() {
        return Some(workspace_path);
    }
    None
}

/// Strict open + full drain: every §1–§4c container invariant holds on
/// the second device profile, at its own parameter values.
#[test]
fn noel_strict_open_drains_2928_pairs_to_the_trailer() {
    let Some(path) = noel_fixture() else {
        eprintln!("skipping noel conformance: noel-son-lumiere.amv not staged");
        return;
    };
    let stream_len = std::fs::metadata(&path).expect("fixture metadata").len();
    let f = File::open(&path).expect("open noel fixture");
    // §2/§3 sentinels must hold under the STRICT open, not just the
    // permissive one — the profile differs only in parameter values.
    let mut d = AmvDemuxer::open_strict(BufReader::new(f)).expect("strict open accepts noel");

    // §2 amvh parameter values from the trace's per-sample columns.
    assert_eq!(d.header().width, 96);
    assert_eq!(d.header().height, 64);
    assert_eq!(d.header().fps, 16);
    assert_eq!(d.header().micros_per_frame, 62_500, "1e6 / 16");
    assert_eq!(d.header().duration_packed, 0x0000_0302, "3:02 as written");

    // §3b audio WAVEFORMATEX — identical across both profiles.
    assert_eq!(d.audio_format().format_tag, 1, "declared PCM (§3b note)");
    assert_eq!(d.audio_format().channels, 1);
    assert_eq!(d.audio_format().samples_per_sec, 22_050);
    assert_eq!(d.audio_format().avg_bytes_per_sec, 44_100);
    assert_eq!(d.audio_format().block_align, 2);
    assert_eq!(d.audio_format().bits_per_sample, 16);

    // Full §4 drain to the trailer.
    loop {
        match d.next_packet() {
            Ok(_) => {}
            Err(Error::Eof) => break,
            Err(e) => panic!("movi walk error: {e:?}"),
        }
    }
    assert_eq!(d.video_frames_emitted(), 2928, "trace: 2928 00dc chunks");
    assert_eq!(d.audio_blocks_emitted(), 2928, "trace: 2928 01wb chunks");
    assert!(d.movi_interleave_balanced(), "§4 strict 1:1 pairing");
    assert!(!d.is_truncated(), "clean trailer-bounded EOF");
    let trailer = d.trailer_offset().expect("AMV_END_ trailer observed");
    assert_eq!(
        trailer + AMV_END_TRAILER.len() as u64,
        stream_len,
        "trailer is the last 8 bytes of the file"
    );
    assert_eq!(d.trailer_matches_eof(stream_len), Some(true));

    // §4b closing observation: this profile's header duration is
    // truncated by one second — the byte-exact boolean check rejects
    // it, the graded check names the device shape.
    assert!(
        !d.duration_consistent_with_drained_frames(),
        "3:02 header vs 2928/16 = 3:03 derivation fails the exact check"
    );
    let grade = d.duration_consistency_with_drained_frames();
    assert_eq!(grade, DurationConsistency::TruncatedByOneSecond);
    assert!(grade.is_device_conformant());
}

/// §4b preamble survey over all 2928 real audio blocks — the byte-level
/// evidence behind the settled one-byte step-index field width, pinned
/// exactly as the trace's survey table records it.
#[test]
fn noel_preamble_survey_matches_the_trace_survey_table() {
    let Some(path) = noel_fixture() else {
        eprintln!("skipping noel preamble survey: noel-son-lumiere.amv not staged");
        return;
    };
    let bytes = std::fs::read(&path).expect("read noel fixture");
    let movi_pos = bytes
        .windows(4)
        .position(|w| w == b"movi")
        .expect("movi FOURCC present");
    let trailer_start = bytes.len() - AMV_END_TRAILER.len();
    let movi_body = &bytes[movi_pos + 4..trailer_start];

    let mut blocks = 0u32;
    let mut nonzero_step = 0u32;
    let mut max_step = 0u8;
    let mut first_step = None;
    for payload in MoviPayloadIter::new(movi_body) {
        let payload = payload.expect("clean chunk walk");
        let MoviPayload::Audio { body, .. } = payload else {
            continue;
        };
        let pre = AmvAudioPreamble::parse(body).expect("8-byte preamble present");
        pre.validate_sentinels().expect("positive sample count");
        // The field-width settler: +0x03 is 0xAA in EVERY block.
        assert_eq!(
            pre.device_constant_byte(),
            0xAA,
            "trace §4b: +3 is 0xAA in all 2928 blocks (block {blocks})"
        );
        // The one-byte step index stays inside the IMA table domain in
        // every block — under the disproven 16-bit reading it would be
        // outside [0, 88] in every one of them.
        assert!(
            pre.step_index_in_ima_range(),
            "block {blocks}: step index {} in [0, {IMA_STEP_INDEX_MAX}]",
            pre.initial_step_index()
        );
        if first_step.is_none() {
            first_step = Some(pre.initial_step_index());
        }
        if pre.initial_step_index() != 0 {
            nonzero_step += 1;
        }
        max_step = max_step.max(pre.initial_step_index());
        blocks += 1;
    }
    assert_eq!(blocks, 2928, "survey covered every 01wb block");
    // Trace survey table row for noel: `+2` first 0, range 0…80,
    // non-zero in 2914 of 2928 blocks.
    assert_eq!(first_step, Some(0), "first block's step index is 0");
    assert_eq!(max_step, 80, "trace: +2 spans 0…80 on this profile");
    assert_eq!(nonzero_step, 2914, "trace: +2 non-zero in 2914 blocks");
}

/// Every one of the 2928 `00dc` payloads satisfies the §4a byte-shape
/// rule (SOI at +0, EOI at end, framing intact) on this profile too.
#[test]
fn noel_every_video_payload_passes_the_4a_shape_check() {
    let Some(path) = noel_fixture() else {
        eprintln!("skipping noel video-shape: noel-son-lumiere.amv not staged");
        return;
    };
    let bytes = std::fs::read(&path).expect("read noel fixture");
    let movi_pos = bytes
        .windows(4)
        .position(|w| w == b"movi")
        .expect("movi FOURCC present");
    let trailer_start = bytes.len() - AMV_END_TRAILER.len();
    let movi_body = &bytes[movi_pos + 4..trailer_start];

    let mut frames = 0u32;
    for payload in MoviPayloadIter::new(movi_body) {
        if let MoviPayload::Video { body, .. } = payload.expect("clean chunk walk") {
            validate_video_payload_shape(body)
                .unwrap_or_else(|e| panic!("frame {frames} fails §4a shape: {e:?}"));
            frames += 1;
        }
    }
    assert_eq!(frames, 2928);
}

/// Seek behaviour holds on the second profile: an indexed mid-stream
/// video seek lands exactly on the requested intra frame and hands back
/// a §4a JPEG payload, matching the linear-walk result.
#[test]
fn noel_indexed_seek_matches_linear_walk() {
    let Some(path) = noel_fixture() else {
        eprintln!("skipping noel seek: noel-son-lumiere.amv not staged");
        return;
    };

    // Linear: drain to video frame 1500 and remember its payload head.
    let f = File::open(&path).expect("open noel fixture");
    let mut linear = AmvDemuxer::open(BufReader::new(f)).expect("open");
    let target_pts = 1500i64;
    let linear_payload = loop {
        let pkt = linear.next_packet().expect("walk to frame 1500");
        if pkt.stream_index == 0 && pkt.pts == Some(target_pts) {
            break pkt.data.clone();
        }
    };

    // Indexed: build the chunk index, seek straight there.
    let f = File::open(&path).expect("open noel fixture");
    let mut indexed = AmvDemuxer::open(BufReader::new(f)).expect("open");
    indexed.build_chunk_index().expect("index build");
    let landed = indexed.seek_to(0, target_pts).expect("indexed seek");
    assert_eq!(landed, target_pts, "intra-only video seeks land exactly");
    let pkt = indexed.next_packet().expect("packet at seek target");
    assert_eq!(pkt.stream_index, 0);
    assert_eq!(pkt.pts, Some(target_pts));
    assert_eq!(&pkt.data[..2], JPEG_SOI, "§4a: payload starts at SOI");
    assert_eq!(pkt.data, linear_payload, "indexed == linear payload bytes");
}
