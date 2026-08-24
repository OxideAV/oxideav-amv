//! Cross-profile **codec** conformance against `noel-son-lumiere.amv`
//! (96 × 64 @ 16 fps): the §4a table-stripped JPEG decode and the §4b
//! IMA-ADPCM decode, established and reference-validated on
//! `comedian.amv`, hold unchanged on the second device profile — per the
//! trace's cross-profile confirmation ("Not one frame of
//! `noel-son-lumiere.amv` needed a different table, a different sampling
//! factor, or a different scan") and its §4b decode table (std 5619,
//! 0.0000 % clip, 4 035 189 samples = 183.00 s).
//!
//! The reference cross-checks use black-box validator binaries only
//! (`djpeg` / `magick` for video, `ffprobe` for audio) and skip when the
//! binary or the fixture is absent; the in-crate decode assertions are
//! fixture-truth and always run when the fixture is staged.

use std::path::{Path, PathBuf};
use std::process::Command;

use oxideav_amv::{
    decode_audio_payload, decode_frame_from_payload, decode_frame_from_payload_with,
    reconstruct_jpeg_from_payload, AmvHeader, ChromaUpsample, MoviPayload, MoviPayloadIter,
    AMVH_BODY_LEN, AMV_END_TRAILER,
};

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

/// Parse the §2 header and split the `movi` body into raw video payloads
/// and raw audio payloads.
fn split_payloads(path: &Path) -> (AmvHeader, Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let bytes = std::fs::read(path).expect("read noel fixture");
    let header =
        AmvHeader::parse(&bytes[0x20..0x20 + AMVH_BODY_LEN as usize]).expect("amvh parses");
    let movi_pos = bytes
        .windows(4)
        .position(|w| w == b"movi")
        .expect("movi FOURCC present");
    let trailer_start = bytes.len() - AMV_END_TRAILER.len();
    let movi_body = &bytes[movi_pos + 4..trailer_start];

    let mut video = Vec::new();
    let mut audio = Vec::new();
    for payload in MoviPayloadIter::new(movi_body) {
        match payload.expect("clean chunk walk") {
            MoviPayload::Video { body, .. } => video.push(body.to_vec()),
            MoviPayload::Audio { body, .. } => audio.push(body.to_vec()),
            other => panic!("noel carries only 00dc/01wb chunks, got {other:?}"),
        }
    }
    (header, video, audio)
}

/// §4b cross-profile audio decode: all 2928 blocks decode to the trace's
/// exact sample total (4 035 189 mono = 183.00 s at 22 050 Hz), with a
/// **completely clip-free** output — the settled step-index-reset rule's
/// signature on this profile (any decode that honoured the preamble
/// `+0x02` field rails at 0.11 %, per the trace's three-rule table).
#[test]
fn noel_audio_decodes_clip_free_to_the_trace_sample_total() {
    let Some(path) = noel_fixture() else {
        eprintln!("skipping noel audio decode: noel-son-lumiere.amv not staged");
        return;
    };
    let (_, _, audio) = split_payloads(&path);
    assert_eq!(audio.len(), 2928);

    let mut pcm: Vec<i16> = Vec::new();
    for payload in &audio {
        pcm.extend(decode_audio_payload(payload).expect("01wb payload decodes"));
    }
    // Trace §4b: "`noel-son-lumiere.amv` 4 035 189 samples = 183.00 s".
    assert_eq!(pcm.len(), 4_035_189, "§4b total decoded sample count");
    let seconds = pcm.len() as f64 / 22_050.0;
    assert!(
        (seconds - 183.0).abs() < 0.01,
        "183.00 s at 22 050 Hz (got {seconds:.4})"
    );

    // Completely clip-free (trace three-rule table: 0.0000 % under the
    // reset rule — the honour-the-field rule rails at 0.1127 %).
    let clipped = pcm
        .iter()
        .filter(|&&s| s == i16::MAX || s == i16::MIN)
        .count();
    assert_eq!(clipped, 0, "reset-rule decode of noel is clip-free");

    // Std ≈ 5619 per the trace's rule table — structured audio, not
    // noise (a wrong-table decode rails toward 9500+).
    let n = pcm.len() as f64;
    let mean = pcm.iter().map(|&s| s as f64).sum::<f64>() / n;
    let var = pcm
        .iter()
        .map(|&s| (s as f64 - mean) * (s as f64 - mean))
        .sum::<f64>()
        / n;
    let std = var.sqrt();
    assert!(
        (5550.0..5700.0).contains(&std),
        "decode std tracks the trace's 5619 (got {std:.0})"
    );
}

/// Black-box audio cross-check: the decoded PCM wrapped in a standard
/// WAV reads back as 22 050 Hz mono ≈ 183.0 s through `ffprobe` (an
/// opaque validator binary; skipped when absent).
#[test]
fn noel_audio_wav_validated_by_ffprobe() {
    let Some(path) = noel_fixture() else {
        eprintln!("skipping noel ffprobe: noel-son-lumiere.amv not staged");
        return;
    };
    if Command::new("ffprobe").arg("-version").output().is_err() {
        eprintln!("skipping noel ffprobe: no ffprobe on PATH");
        return;
    }
    let (_, _, audio) = split_payloads(&path);
    let mut pcm: Vec<i16> = Vec::new();
    for payload in &audio {
        pcm.extend(decode_audio_payload(payload).expect("01wb payload decodes"));
    }

    // Standard 44-byte-header mono 16-bit WAV.
    let sample_rate = 22_050u32;
    let data_len = (pcm.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + pcm.len() * 2);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for &s in &pcm {
        wav.extend_from_slice(&s.to_le_bytes());
    }
    let dir = std::env::temp_dir().join("oxideav_amv_noel_codec");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let wav_path = dir.join("noel_audio.wav");
    std::fs::write(&wav_path, &wav).expect("write wav");

    let probe = |entry: &str| -> String {
        let out = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                entry,
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                wav_path.to_str().unwrap(),
            ])
            .output()
            .expect("ffprobe runs");
        assert!(out.status.success(), "ffprobe accepted the WAV");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    assert_eq!(probe("stream=sample_rate"), "22050");
    assert_eq!(probe("stream=channels"), "1");
    let dur: f64 = probe("stream=duration").parse().expect("numeric duration");
    assert!(
        (dur - 183.0).abs() < 0.05,
        "independent read-back ≈ 183.0 s (got {dur})"
    );
}

/// §4a cross-profile video decode: **all 2928** frames decode in-crate
/// at 96 × 64 with the identical device tables — the trace's zero-
/// failure cross-profile confirmation, reproduced through this crate's
/// own decoder with no external binary.
#[test]
fn noel_all_2928_frames_decode_in_crate() {
    let Some(path) = noel_fixture() else {
        eprintln!("skipping noel all-frames decode: noel-son-lumiere.amv not staged");
        return;
    };
    let (header, video, _) = split_payloads(&path);
    assert_eq!(video.len(), 2928);
    assert_eq!((header.width, header.height), (96, 64));

    let mut coherent = 0u32;
    for (i, payload) in video.iter().enumerate() {
        let frame = decode_frame_from_payload(&header, payload)
            .unwrap_or_else(|e| panic!("frame {i} fails in-crate decode: {e:?}"));
        assert_eq!((frame.width, frame.height), (96, 64), "frame {i} geometry");
        assert_eq!(frame.rgb.len(), 96 * 64 * 3, "frame {i} raster size");

        // Natural-content oracle (as in the comedian all-frames test):
        // luma std well above flat noise floor counts as coherent
        // content; fade/black frames are allowed but must be a small
        // minority across a 3-minute natural clip.
        let mut mean = 0.0f64;
        for px in frame.rgb.chunks_exact(3) {
            mean += 0.299 * px[0] as f64 + 0.587 * px[1] as f64 + 0.114 * px[2] as f64;
        }
        mean /= (96 * 64) as f64;
        let mut var = 0.0f64;
        for px in frame.rgb.chunks_exact(3) {
            let y = 0.299 * px[0] as f64 + 0.587 * px[1] as f64 + 0.114 * px[2] as f64;
            var += (y - mean) * (y - mean);
        }
        let std = (var / (96 * 64) as f64).sqrt();
        if std > 4.0 {
            coherent += 1;
        }
    }
    assert!(
        coherent >= 2928 * 9 / 10,
        "overwhelming majority of frames carry coherent content ({coherent}/2928)"
    );
}

/// Black-box video cross-check on sampled frames: the in-crate decode
/// tracks a reference JPEG decoder's pixels on the reconstructed frames
/// within the same error envelope measured on comedian (Nearest ≈ 1.35,
/// Triangle ≈ 0.05 MAE/channel). Skipped when no decoder binary is on
/// `PATH`; the validator is an opaque process, no decoder source is
/// read.
#[test]
fn noel_sampled_frames_match_black_box_reference_decoder() {
    let Some(path) = noel_fixture() else {
        eprintln!("skipping noel reference decode: noel-son-lumiere.amv not staged");
        return;
    };
    let use_djpeg = Command::new("djpeg").arg("-help").output().is_ok();
    let use_magick = !use_djpeg && Command::new("magick").arg("--version").output().is_ok();
    if !use_djpeg && !use_magick {
        eprintln!("skipping noel reference decode: no djpeg/magick on PATH");
        return;
    }
    let (header, video, _) = split_payloads(&path);

    let dir = std::env::temp_dir().join("oxideav_amv_noel_codec");
    std::fs::create_dir_all(&dir).expect("temp dir");

    // The trace's cross-profile subsampling table samples frames
    // 0 / 5 / 500; add a late frame for coverage.
    for &idx in &[0usize, 5, 500, 2900] {
        let jpeg = reconstruct_jpeg_from_payload(&header, &video[idx])
            .expect("frame reconstructs to a conforming JPEG");
        let in_path = dir.join("ref_frame.jpg");
        std::fs::write(&in_path, &jpeg).expect("write jpeg");
        let out = if use_djpeg {
            Command::new("djpeg")
                .args(["-pnm", in_path.to_str().unwrap()])
                .output()
        } else {
            Command::new("magick")
                .args([in_path.to_str().unwrap(), "ppm:-"])
                .output()
        }
        .expect("validator runs");
        assert!(
            out.status.success(),
            "frame {idx}: reference decoder consumed the reconstruction cleanly \
             (the §4a no-premature-end oracle): {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let ppm = out.stdout;
        // Minimal P6 parse: "P6\n<w> <h>\n255\n" + raw RGB. The
        // reconstruction is upright only after the §4a flip, which the
        // in-crate decode applies — the reference decodes the JPEG's
        // coded (bottom-up) order, so flip its rows for comparison.
        let header_end = ppm
            .windows(4)
            .position(|w| w == b"255\n")
            .expect("PPM maxval")
            + 4;
        let mut ref_rgb = ppm[header_end..].to_vec();
        assert_eq!(ref_rgb.len(), 96 * 64 * 3, "frame {idx}: reference raster");
        oxideav_amv::flip_rows_vertical(&mut ref_rgb, 64, 96 * 3);

        for (ups, bound) in [
            (ChromaUpsample::Nearest, 2.5f64),
            (ChromaUpsample::Triangle, 0.8f64),
        ] {
            let ours =
                decode_frame_from_payload_with(&header, &video[idx], ups).expect("in-crate decode");
            let mae = ours
                .rgb
                .iter()
                .zip(&ref_rgb)
                .map(|(&a, &b)| (a as f64 - b as f64).abs())
                .sum::<f64>()
                / ref_rgb.len() as f64;
            assert!(
                mae < bound,
                "frame {idx}: MAE {mae:.3} within {bound} for {ups:?}"
            );
        }
    }
}
