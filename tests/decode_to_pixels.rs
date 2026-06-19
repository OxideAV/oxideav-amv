//! End-to-end milestone validation: a real AMV `00dc` video frame, run
//! through the §4a device-stripped-JPEG reconstruction
//! ([`oxideav_amv::reconstruct_jpeg_from_payload`]), decodes to a
//! coherent pixel raster.
//!
//! The crate is a *container*: it reassembles the marker segments the
//! AMV device strips (`DQT` / `SOF0` / `DHT` / `SOS`) and copies the
//! entropy-coded scan through verbatim, producing a standards-conforming
//! baseline JFIF/JPEG. The heavyweight DCT / Huffman image decode is the
//! downstream `mjpeg` codec's job — so this test does not implement a
//! decoder. It proves the reconstruction is *decodable to pixels* by
//! handing the reconstructed bytes to a **black-box JPEG decoder binary**
//! (`djpeg` from libjpeg, or `magick`/`ffmpeg`) exactly as trace §4a's
//! reconstruction oracle prescribes:
//!
//! > "A reconstruction is judged correct when (a) a strict baseline
//! > decoder consumes the entropy stream with no 'premature end of data'
//! > error … and (b) the decoded raster is a coherent natural image."
//!
//! No decoder *source* is read — the validator is an opaque process that
//! consumes the reconstructed JPEG and emits pixels; this test only
//! inspects those pixels. Skipped automatically when no JPEG decoder
//! binary is on `PATH` (CI ships libjpeg; some dev machines may not).
//!
//! ## What is asserted (trace §4a)
//!
//! * **Clean decode** — the validator exits successfully (no premature
//!   end of data), i.e. the hardcoded MCU geometry (4:2:0, 6 blocks/MCU)
//!   exactly matches the bit budget of the verbatim Annex-K Huffman
//!   tables. Wrong tables/sampling would desync and fail here.
//! * **Geometry** — the decoded raster is the §2 `amvh` resolution
//!   (128 × 96 for `comedian.amv`).
//! * **Coherent natural image** — the luma plane has a real tonal range
//!   (std well above flat-noise) and *low* vertical total-variation
//!   (smooth, not the scrambled 4-wide strips a wrong 4:1:1 sampling
//!   would produce). Trace §4a reports a vert-TV near 8.8 for frame 0
//!   under 4:2:0; we assert a generous upper bound that 4:1:1 / noise
//!   would blow past.
//! * **Inter-frame consistency** — three reconstructed frames decode to
//!   the same geometry and a stable tonal profile (an intra-only,
//!   fixed-table codec — every frame is a keyframe, §4a).

use std::path::{Path, PathBuf};
use std::process::Command;

use oxideav_amv::{
    reconstruct_jpeg_from_payload, AmvHeader, MoviPayload, MoviPayloadIter, AMVH_BODY_LEN,
    AMV_END_TRAILER,
};

/// Locate the staged `comedian.amv` fixture, mirroring the in-crate unit
/// tests: prefer a crate-local copy, fall back to the docs submodule.
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

/// Parse the §2 `amvh` header and the first `n` reconstructed JPEG frames
/// of `comedian.amv`.
fn reconstruct_first_frames(path: &Path, n: usize) -> (AmvHeader, Vec<Vec<u8>>) {
    let bytes = std::fs::read(path).expect("read comedian fixture");
    let header =
        AmvHeader::parse(&bytes[0x20..0x20 + AMVH_BODY_LEN as usize]).expect("amvh parses");

    let movi_pos = bytes
        .windows(4)
        .position(|w| w == b"movi")
        .expect("movi FOURCC present");
    let trailer_start = bytes.len() - AMV_END_TRAILER.len();
    let movi_body = &bytes[movi_pos + 4..trailer_start];

    let mut frames = Vec::new();
    for payload in MoviPayloadIter::new(movi_body).filter_map(|r| r.ok()) {
        if let MoviPayload::Video { body, .. } = payload {
            let jpeg = reconstruct_jpeg_from_payload(&header, body)
                .expect("real frame reconstructs to a conforming JPEG");
            frames.push(jpeg);
            if frames.len() == n {
                break;
            }
        }
    }
    (header, frames)
}

/// A decoded 8-bit RGB raster.
struct Raster {
    width: usize,
    height: usize,
    /// Interleaved R,G,B bytes, row-major top-to-bottom (PPM order).
    rgb: Vec<u8>,
}

/// Parse a binary `P6` PPM (the format `djpeg -pnm` and `magick … ppm:-`
/// both emit for an RGB image).
fn parse_ppm_p6(data: &[u8]) -> Option<Raster> {
    if data.len() < 2 || &data[0..2] != b"P6" {
        return None;
    }
    let mut idx = 2usize;
    let mut tokens = Vec::with_capacity(3);
    while tokens.len() < 3 {
        // Skip whitespace + comments.
        while idx < data.len() && (data[idx] as char).is_whitespace() {
            idx += 1;
        }
        if idx < data.len() && data[idx] == b'#' {
            while idx < data.len() && data[idx] != b'\n' {
                idx += 1;
            }
            continue;
        }
        let start = idx;
        while idx < data.len() && !(data[idx] as char).is_whitespace() {
            idx += 1;
        }
        let tok: usize = std::str::from_utf8(&data[start..idx]).ok()?.parse().ok()?;
        tokens.push(tok);
    }
    // Single whitespace byte separates the header from the pixel data.
    idx += 1;
    let (w, h, maxval) = (tokens[0], tokens[1], tokens[2]);
    if maxval != 255 {
        return None;
    }
    let need = w * h * 3;
    if data.len() < idx + need {
        return None;
    }
    Some(Raster {
        width: w,
        height: h,
        rgb: data[idx..idx + need].to_vec(),
    })
}

/// BT.601 luma of each pixel.
fn luma_plane(r: &Raster) -> Vec<f64> {
    (0..r.width * r.height)
        .map(|k| {
            let p = &r.rgb[k * 3..k * 3 + 3];
            0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64
        })
        .collect()
}

/// `(mean, std)` of a slice.
fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Mean absolute luma difference between vertically adjacent rows — the
/// §4a "coherence" oracle. A coherent natural raster is smooth (low TV);
/// the wrong-sampling (4:1:1) reconstruction scrambles luma blocks into
/// strips and inflates this sharply.
fn vertical_total_variation(luma: &[f64], width: usize, height: usize) -> f64 {
    let mut acc = 0.0f64;
    let mut n = 0u64;
    for row in 0..height.saturating_sub(1) {
        for col in 0..width {
            acc += (luma[row * width + col] - luma[(row + 1) * width + col]).abs();
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        acc / n as f64
    }
}

// --- black-box JPEG decoder discovery ---------------------------------

/// A decoder kind we know how to drive to an RGB `P6` PPM on stdout.
#[derive(Clone, Copy)]
enum JpegDecoder {
    /// libjpeg's `djpeg` — the strict baseline decoder the §4a oracle
    /// names. `djpeg -pnm <file>` writes a `P6` PPM to stdout and fails
    /// (non-zero exit) on a truncated / premature-end scan.
    Djpeg,
    /// ImageMagick `magick <file> ppm:-`.
    Magick,
}

/// Whether spawning `name <probe_arg>` succeeds at all (the process
/// launched — exit status is irrelevant; `djpeg` exits non-zero for a
/// help/version probe on some builds, but a spawn error means the binary
/// is absent from `PATH`).
fn binary_present(name: &str, probe_arg: &str) -> bool {
    Command::new(name).arg(probe_arg).output().is_ok()
}

/// Pick the first available black-box decoder, preferring the strict
/// `djpeg` (its clean exit is itself the "no premature end of data"
/// signal §4a relies on).
fn find_decoder() -> Option<JpegDecoder> {
    if binary_present("djpeg", "-help") {
        return Some(JpegDecoder::Djpeg);
    }
    if binary_present("magick", "--version") {
        return Some(JpegDecoder::Magick);
    }
    None
}

/// Decode `jpeg` to a `P6` PPM raster via the chosen black-box binary.
/// Returns `Err` (with the validator's stderr) on a non-clean decode.
fn decode_to_raster(dec: JpegDecoder, jpeg: &[u8]) -> Result<Raster, String> {
    let dir = std::env::temp_dir().join("oxideav_amv_decode_to_pixels");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let in_path = dir.join("frame.jpg");
    std::fs::write(&in_path, jpeg).map_err(|e| e.to_string())?;

    let out = match dec {
        JpegDecoder::Djpeg => Command::new("djpeg")
            .args(["-pnm", in_path.to_str().unwrap()])
            .output(),
        JpegDecoder::Magick => Command::new("magick")
            .args([in_path.to_str().unwrap(), "ppm:-"])
            .output(),
    }
    .map_err(|e| e.to_string())?;

    if !out.status.success() {
        return Err(format!(
            "decoder exited with {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    parse_ppm_p6(&out.stdout).ok_or_else(|| "validator did not emit a P6 PPM".to_string())
}

#[test]
fn comedian_frame_reconstructs_and_decodes_to_coherent_pixels() {
    let Some(path) = comedian_fixture() else {
        eprintln!("skipping decode-to-pixels: comedian.amv not staged");
        return;
    };
    let Some(dec) = find_decoder() else {
        eprintln!("skipping decode-to-pixels: no djpeg/magick on PATH");
        return;
    };

    let (header, frames) = reconstruct_first_frames(&path, 3);
    assert_eq!(frames.len(), 3, "three reconstructed video frames");
    assert_eq!((header.width, header.height), (128, 96), "§2 amvh geometry");

    let mut profiles = Vec::new();
    for (i, jpeg) in frames.iter().enumerate() {
        // (a) Clean decode — no premature end of data. This is the §4a
        // bit-budget oracle: a desynced reconstruction fails here.
        let raster = match decode_to_raster(dec, jpeg) {
            Ok(r) => r,
            Err(e) => panic!("frame {i} reconstruction did not decode cleanly: {e}"),
        };

        // Geometry equals the §2 amvh resolution.
        assert_eq!(
            (raster.width, raster.height),
            (128, 96),
            "frame {i} decoded raster geometry"
        );

        // (b) Coherent natural image.
        let luma = luma_plane(&raster);
        let (mean, std) = mean_std(&luma);
        let vtv = vertical_total_variation(&luma, raster.width, raster.height);
        eprintln!("frame {i}: lumaMean={mean:.1} lumaStd={std:.1} vertTV={vtv:.2}");

        // Real tonal range — not a flat / single-colour decode (which a
        // catastrophic table mismatch can also yield without erroring).
        assert!(
            std > 10.0,
            "frame {i} luma std {std:.1} too flat to be a natural image"
        );
        // Smoothness — 4:2:0 keeps vert-TV low (§4a ≈ 8.8). The wrong
        // 4:1:1 sampling visibly scrambles luma blocks and pushes this up;
        // a noise-like decode pushes it far higher still. 25.0 is a
        // generous bound that a coherent raster clears comfortably and a
        // scrambled one does not.
        assert!(
            vtv < 25.0,
            "frame {i} vertical total-variation {vtv:.2} too high for a coherent 4:2:0 raster"
        );

        profiles.push((mean, std));
    }

    // (c) Inter-frame consistency: an intra-only fixed-table codec decodes
    // every frame against the same hardcoded tables, so the tonal profile
    // is stable frame-to-frame (consecutive frames of a single shot).
    let (m0, _) = profiles[0];
    for (i, &(mean, _)) in profiles.iter().enumerate() {
        assert!(
            (mean - m0).abs() < 40.0,
            "frame {i} luma mean {mean:.1} drifted unexpectedly from frame 0 {m0:.1}"
        );
    }
}

/// §4a bottom-up orientation, end-to-end: the baseline-JPEG decode of the
/// reconstructed frame comes out vertically mirrored (DIB row order), and
/// the public [`oxideav_amv::flip_rows_vertical`] blit-time transform is
/// the documented correction. Applies it to a *real* decoded raster and
/// pins the §4a involution property on real pixels.
#[test]
fn comedian_frame_vertical_flip_yields_upright_and_round_trips() {
    let Some(path) = comedian_fixture() else {
        eprintln!("skipping orientation: comedian.amv not staged");
        return;
    };
    let Some(dec) = find_decoder() else {
        eprintln!("skipping orientation: no djpeg/magick on PATH");
        return;
    };

    let (_header, frames) = reconstruct_first_frames(&path, 1);
    let raster = match decode_to_raster(dec, &frames[0]) {
        Ok(r) => r,
        Err(e) => panic!("frame reconstruction did not decode cleanly: {e}"),
    };
    let bytes_per_row = raster.width * 3; // P6 = 3 bytes/pixel.
    let mirrored = raster.rgb.clone();

    // The §4a flip changes a non-degenerate natural raster (the decode is
    // not accidentally top-bottom symmetric).
    let mut upright = mirrored.clone();
    oxideav_amv::flip_rows_vertical(&mut upright, raster.height, bytes_per_row);
    assert_ne!(
        upright, mirrored,
        "vertical flip must change the mirrored decode output"
    );

    // Top row of the upright raster equals the bottom row of the mirrored
    // decode (DIB bottom-up convention) — the literal §4a row reversal.
    let last_row = (raster.height - 1) * bytes_per_row;
    assert_eq!(
        &upright[..bytes_per_row],
        &mirrored[last_row..last_row + bytes_per_row],
        "upright top row == mirrored bottom row"
    );

    // Involution: flipping the upright raster again returns the decoder's
    // mirrored output verbatim.
    oxideav_amv::flip_rows_vertical(&mut upright, raster.height, bytes_per_row);
    assert_eq!(upright, mirrored, "double flip restores the decode output");
}
