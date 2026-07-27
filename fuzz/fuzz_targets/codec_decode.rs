#![no_main]

//! Drive arbitrary fuzz-supplied bytes through the two **codec-level**
//! decode paths — the §4a table-stripped baseline-JPEG frame decode and
//! the §4b IMA-ADPCM block decode — plus the rate-controlled §4a
//! *encode* on fuzz-derived pixel content.
//!
//! Where `parse.rs` covers the byte parsers and `demuxer_open.rs` the
//! container walk, this target covers the entropy-level machinery those
//! layers hand payloads to:
//!
//! 1. [`oxideav_amv::decode_frame_from_payload`] — the full §4a decode
//!    stack (byte-destuffing bit reader, canonical Huffman walk,
//!    dequant, IDCT, 4:2:0 upsample, colour convert, crop + flip) with
//!    the frame geometry taken from attacker-controlled header fields,
//!    exactly as a hostile `.amv` would supply them. Geometry is capped
//!    to 256×256 by masking so a single iteration cannot allocate
//!    unbounded planes from a 4-byte claim (the demuxer's own header
//!    path is fuzzed separately; this target aims at the entropy walk).
//! 2. [`oxideav_amv::decode_frame_yuv420p_from_payload`] — the native
//!    planar variant over the same bytes (shares the entropy walk but
//!    exercises the plane-crop path).
//! 3. [`oxideav_amv::decode_audio_payload`] — the §4b preamble parse +
//!    nibble decode, whose `decoded_sample_count` dword is fully
//!    attacker-controlled and must never drive an allocation beyond the
//!    actual nibble budget of the supplied body.
//! 4. [`oxideav_amv::encode_frame_rgb_with_budget`] — the §4a budgeted
//!    encode over pixels expanded from the fuzz input, at a budget the
//!    input's first bytes choose. The trim search must terminate and
//!    return for any content/budget combination; its output must
//!    round-trip through `decode_frame_from_payload` without error
//!    (encode → decode of a *valid* frame is a total function).
//!
//! Contract under test: every call returns a `Result` — no panic, no
//! debug-build integer overflow, no out-of-bounds index, no allocation
//! proportional to an attacker-claimed field. Return values are
//! intentionally discarded except where the encode→decode round-trip
//! asserts success on known-valid input.

use libfuzzer_sys::fuzz_target;
use oxideav_amv::{
    decode_audio_payload, decode_frame_from_payload, decode_frame_from_payload_with,
    decode_frame_yuv420p_from_payload, encode_frame_rgb_with_budget, AmvHeader, ChromaUpsample,
};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    // Attacker-chosen geometry, masked to keep one iteration's plane
    // allocations bounded (1..=256 on each axis).
    let width = (u16::from_le_bytes([data[0], data[1]]) & 0xFF) as u32 + 1;
    let height = (u16::from_le_bytes([data[2], data[3]]) & 0xFF) as u32 + 1;
    let budget = u16::from_le_bytes([data[4], data[5]]) as usize;
    let body = &data[6..];

    let header = AmvHeader {
        micros_per_frame: 83_333,
        width,
        height,
        fps: 12,
        flag_one: 1,
        reserved_30: 0,
        duration_packed: 0,
    };

    // 1 + 2: hostile bytes into both §4a decode front doors — the RGB
    // path under both chroma-upsampling filters (the triangle filter's
    // clamped edge taps must stay inside the visible plane for every
    // hostile geometry) and the native planar path.
    let _ = decode_frame_from_payload(&header, body);
    let _ = decode_frame_from_payload_with(&header, body, ChromaUpsample::Triangle);
    let _ = decode_frame_yuv420p_from_payload(&header, body);

    // 3: hostile bytes into the §4b audio decode.
    let _ = decode_audio_payload(body);

    // 4: budgeted encode on fuzz-derived pixels (tile the body across
    // the raster), then the round-trip that must always succeed.
    let (w, h) = (width.min(64), height.min(64));
    let n = (w * h * 3) as usize;
    let mut rgb = vec![0u8; n];
    for (i, px) in rgb.iter_mut().enumerate() {
        *px = body[i % body.len()];
    }
    let frame = encode_frame_rgb_with_budget(w, h, &rgb, budget)
        .expect("budgeted encode of a valid raster is total");
    decode_frame_from_payload(
        &AmvHeader {
            width: w,
            height: h,
            ..header
        },
        &frame.payload,
    )
    .expect("encoder output must always decode");
});
