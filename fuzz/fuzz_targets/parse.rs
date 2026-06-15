#![no_main]

//! Drive arbitrary fuzz-supplied bytes through every public AMV byte
//! parser and payload-shape validator.
//!
//! The trace document records every parse entry point as a pure
//! `bytes → Result<…, AmvDemuxerError>` function — there are no
//! background tasks, no inner unwraps, no `Vec` pre-allocations
//! proportional to an attacker-supplied header field. The contract
//! under test is that each call **returns** for any byte sequence:
//! a well-formed input yields `Ok(…)` (e.g. `AmvHeader::parse` on a
//! 56-byte body, `AmvWaveFormat::parse` on an 18+ byte body), and a
//! malformed one yields `Err(AmvDemuxerError::InvalidData(_))`.
//! Neither branch may panic, integer-overflow in a debug build, or
//! pre-allocate beyond `data.len()`.
//!
//! The entry points exercised are:
//!
//! 1. [`oxideav_amv::AmvHeader::parse`] — the §2 56-byte `amvh` body
//!    `[µs/frame, w, h, fps, flag_one, reserved_30, packed_duration]`.
//! 2. [`oxideav_amv::AmvHeader::validate_sentinels`] — the strict §2
//!    cross-check (`micros_per_frame == 1_000_000 / fps`, `flag_one
//!    == 1`, `reserved_30 == 0`).
//! 3. [`oxideav_amv::AmvWaveFormat::parse`] — the §3b 20-byte
//!    `WAVEFORMATEX` body (channels, sample rate, byte rate, block
//!    align, bps, cbSize).
//! 4. [`oxideav_amv::AmvWaveFormat::validate_sentinels`] — the strict
//!    §3b cross-check (PCM tag / mono / `avgBytes == samples * 2` /
//!    `blockAlign == 2` / `bps == 16` / `cbSize == 0`).
//! 5. [`oxideav_amv::AmvWaveFormat::frame_interval_samples`] — the
//!    §4b worked-example sample budget (`samples_per_sec / fps`,
//!    integer truncation), driven against an attacker-chosen `fps`
//!    extracted from the first 4 input bytes.
//! 6. [`oxideav_amv::AmvDuration::from_packed`] +
//!    [`oxideav_amv::AmvDuration::to_packed`] — the §2 `+0x34`
//!    `[seconds, minutes, hours, 0]` round-trip.
//! 7. [`oxideav_amv::AmvDuration::from_frame_count`] +
//!    [`oxideav_amv::AmvDuration::is_consistent_with_frame_count`] —
//!    the §2 `frame_count / fps` derivation and its cross-check.
//! 8. [`oxideav_amv::ChunkHeader::parse`] — the §4 8-byte leaf
//!    `[FOURCC, size]` header.
//! 9. [`oxideav_amv::AmvAudioPreamble::parse`] +
//!    [`oxideav_amv::AmvAudioPreamble::validate_sentinels`] +
//!    [`oxideav_amv::AmvAudioPreamble::is_consistent_with_frame_interval`] +
//!    [`oxideav_amv::AmvAudioPreamble::nibble_body_len`] +
//!    [`oxideav_amv::AmvAudioPreamble::is_consistent_with_body_len`] +
//!    [`oxideav_amv::AmvAudioPreamble::body_padding_slack`]
//!    — the §4b 8-byte preamble + its strict / cross-check helpers.
//! 10. [`oxideav_amv::validate_video_payload_shape`] +
//!     [`oxideav_amv::validate_video_payload_no_internal_markers`]
//!     — the §4a `00dc` payload sentinels (SOI / EOI bracket + no
//!     internal JPEG markers).
//!
//! Every entry point takes the **same** fuzz buffer so libfuzzer's
//! corpus minimisation can converge on inputs that exercise every
//! shape simultaneously. The return values are intentionally
//! discarded — this target is panic-free-only by design.

use libfuzzer_sys::fuzz_target;
use oxideav_amv::{
    validate_video_payload_no_internal_markers, validate_video_payload_shape, AmvAudioPreamble,
    AmvDuration, AmvHeader, AmvWaveFormat, ChunkHeader,
};

fuzz_target!(|data: &[u8]| {
    // -----------------------------------------------------------------
    // §2 amvh body parsers + strict cross-check.
    // -----------------------------------------------------------------
    if let Ok(header) = AmvHeader::parse(data) {
        // `validate_sentinels` returns `Err(InvalidData)` for the vast
        // majority of fuzz inputs but the contract is "no panic on any
        // header value", so we drive it on every successful parse.
        let _ = header.validate_sentinels();
        // The duration accessors are infallible.
        let _ = header.duration();
        let _ = header.duration_micros();
    }

    // -----------------------------------------------------------------
    // §3b WAVEFORMATEX parser + strict cross-check + §4b sample budget.
    // -----------------------------------------------------------------
    if let Ok(wf) = AmvWaveFormat::parse(data) {
        let _ = wf.validate_sentinels();
        // Drive `frame_interval_samples` against an attacker-chosen fps
        // pulled from the first four input bytes. This stresses the
        // `samples_per_sec / fps` integer truncation across the entire
        // u32 fps range, including the documented zero-fps guard.
        if data.len() >= 4 {
            let fps = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            let _ = wf.frame_interval_samples(fps);
        }
    }

    // -----------------------------------------------------------------
    // §2 packed-duration round-trip + frame-count derivation.
    // -----------------------------------------------------------------
    if data.len() >= 4 {
        let packed = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let dur = AmvDuration::from_packed(packed);
        // `to_packed` is the documented inverse — round-trip it.
        let _ = dur.to_packed();
        let _ = dur.total_seconds();
    }
    if data.len() >= 12 {
        // Attacker-chosen `(frame_count, fps)` pair. The function is
        // documented infallible (saturates components at u8::MAX,
        // short-circuits on fps == 0) so any byte combination is fair
        // game.
        let frame_count = u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        let fps = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let derived = AmvDuration::from_frame_count(frame_count, fps);
        let _ = derived.to_packed();
        let _ = derived.is_consistent_with_frame_count(frame_count, fps);
    }

    // -----------------------------------------------------------------
    // §4 chunk-header parser. The header is just `[FOURCC, size]` so
    // every 8+ byte slice is a candidate; the parser must never
    // allocate proportionally to the claimed `size`, only report it.
    // -----------------------------------------------------------------
    let _ = ChunkHeader::parse(data);
    if let Ok(hdr) = ChunkHeader::parse(data) {
        let _ = hdr.advance_total();
        let _ = hdr.kind();
    }

    // -----------------------------------------------------------------
    // §4b audio preamble parsers + strict / cross-check.
    // -----------------------------------------------------------------
    if let Ok(preamble) = AmvAudioPreamble::parse(data) {
        let _ = preamble.validate_sentinels();
        // §4b nibble budget: the expected compressed-body byte count and
        // the full-payload cross-check must never overflow or panic on an
        // attacker-chosen decoded_sample_count.
        let _ = preamble.nibble_body_len();
        let _ = preamble.is_consistent_with_body_len(data.len() as u64);
        // §4b padding slack: must never overflow / panic on an
        // attacker-chosen total length vs. decoded_sample_count.
        let _ = preamble.body_padding_slack(data.len() as u64);
        // Cross-check against an attacker-chosen rate / fps pair.
        if data.len() >= 16 {
            let rate = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
            let fps = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
            let _ = preamble.is_consistent_with_frame_interval(rate, fps);
            // Drive the body-length cross-check with an attacker-chosen
            // total length too (independent of the actual slice length).
            let total = u64::from_le_bytes([
                data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
            ]);
            let _ = preamble.is_consistent_with_body_len(total);
            let _ = preamble.body_padding_slack(total);
        }
    }

    // -----------------------------------------------------------------
    // §4a video payload sentinels. Both functions accept any byte
    // slice (4-byte minimum enforced inside the helper) and must never
    // panic regardless of the SOI / EOI / interior-marker shape.
    // -----------------------------------------------------------------
    let _ = validate_video_payload_shape(data);
    let _ = validate_video_payload_no_internal_markers(data);
});
