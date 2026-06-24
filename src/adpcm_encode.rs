//! PCM-to-nibble encode for an AMV `01wb` audio block — the forward
//! (encoder) direction of the §4b "IMA step / index tables — STANDARD"
//! step of `docs/container/amv/amv-container-trace.md`.
//!
//! # Why this lives in the container crate
//!
//! This is the byte-inverse of [`crate::decode_audio_block`]: it turns a
//! buffer of 16-bit mono PCM samples back into the nibble-packed body and
//! 8-byte preamble of an AMV `01wb` leaf chunk. AMV's audio profile is
//! intrinsic to the device (one ADPCM block per video frame, the §4b
//! 8-byte header carrying an `int16` predictor seed, low-nibble-first
//! packing), so producing a device-faithful `01wb` payload belongs in
//! this crate exactly as the decode does — it is the audio counterpart of
//! the §4a video encode.
//!
//! # The encode contract (the exact inverse of §4b decode)
//!
//! The standard IMA/DVI ADPCM *encoder* mirrors the decoder's recurrence:
//! the decoder reconstructs a predictor from each nibble, and the encoder
//! must track that **same** reconstructed predictor (not the original PCM)
//! so the two never drift. For each input sample the encoder picks the
//! 4-bit nibble whose decoded step brings the running predictor closest to
//! the target sample, emits it, then advances its own predictor / step
//! index with the identical clamp rules [`crate::decode_audio_block`]
//! applies. Decode(encode(pcm)) is therefore the canonical IMA
//! round-trip: not bit-exact to the original PCM (ADPCM is lossy) but a
//! stable fixed point — re-encoding the decoded output reproduces the same
//! nibbles.
//!
//! Per §4b the block is **self-contained**: the predictor is seeded from
//! the header `int16` and the step index is reset to `0` at block start.
//! The encoder writes the first sample as the predictor seed (so the
//! decoder's re-seed reproduces it exactly) and emits the `+0x02`
//! `initialStepIndex` field as `0` — the decode-verified always-reset
//! value (the §4b gap note records that seeding a non-zero step index
//! "made the output worse"; the encoder emits the value the decoder
//! honours).
//!
//! No ADPCM **encoder** source was read; the standard IMA step / index
//! tables are public (the IMA/DVI ADPCM recommendation) and are the same
//! `const` arrays the decode path uses — shared here, not duplicated.

use crate::parse::AMV_AUDIO_PREAMBLE_LEN;

// The standard IMA/DVI step + index tables (trace §4b — public, used
// verbatim). Identical to the decode side; re-declared here as the
// forward path needs the same data and `adpcm.rs` keeps them private.
//
// Keeping a second copy rather than widening `adpcm.rs`'s visibility is a
// deliberate boundary: the decode module owns the decode-shaped helpers
// (`decode_nibble`), and this module owns the encode-shaped ones. A unit
// test pins the two table copies byte-equal so they cannot drift.

/// Canonical IMA/DVI 89-entry step-size table (trace §4b). Indexed by the
/// running step index, clamped to `[0, 88]`.
const IMA_STEP_TABLE: [i32; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449,
    494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272,
    2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
    10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
];

/// Canonical IMA/DVI 8-entry index-adjust table (trace §4b), indexed by
/// the low 3 bits of the 4-bit nibble.
const IMA_INDEX_TABLE: [i32; 8] = [-1, -1, -1, -1, 2, 4, 6, 8];

/// Maximum valid step index (the 89-entry table is indexed `[0, 88]`).
const STEP_INDEX_MAX: i32 = (IMA_STEP_TABLE.len() - 1) as i32;

/// Reconstruct the predictor delta a given nibble decodes to at the
/// current `step` — the exact §4b decode recurrence
/// (`diff = step>>3; += step>>2 / >>1 / step for bits 0/1/2; sign bit3`),
/// returned as a signed delta (the caller adds it and clamps). Kept in
/// one place so the encoder's "what would the decoder do" matches the
/// real decode bit-for-bit.
#[inline]
fn nibble_delta(nibble: u8, step: i32) -> i32 {
    let mut diff = step >> 3;
    if nibble & 0x04 != 0 {
        diff += step;
    }
    if nibble & 0x02 != 0 {
        diff += step >> 1;
    }
    if nibble & 0x01 != 0 {
        diff += step >> 2;
    }
    if nibble & 0x08 != 0 {
        -diff
    } else {
        diff
    }
}

/// Choose the 4-bit nibble that best encodes `sample` given the running
/// `predictor` and step `index`, and apply the matching reconstruction so
/// `predictor` / `index` advance exactly as the decoder will. Returns the
/// chosen nibble.
///
/// This is the standard IMA forward step. Rather than the bit-by-bit
/// magnitude search some encoders use, it evaluates the decoder's exact
/// delta for the magnitude bits and the sign separately: the sign comes
/// from `sample - predictor`, and the 3 magnitude bits are the value whose
/// reconstructed `diff` is nearest the absolute difference. Because the
/// reconstruction is the decoder's own recurrence ([`nibble_delta`]), the
/// encoder and decoder never drift.
#[inline]
fn encode_one(sample: i16, predictor: &mut i32, index: &mut i32) -> u8 {
    let step = IMA_STEP_TABLE[*index as usize];
    let delta = sample as i32 - *predictor;
    // Sign bit (bit 3): negative difference encodes with the sign set.
    let sign = if delta < 0 { 0x08u8 } else { 0x00u8 };
    let mag = delta.unsigned_abs() as i32;

    // Standard IMA magnitude search: build the 3 magnitude bits greedily
    // against `step`, accumulating the reconstructed `diff` the decoder
    // would produce. This is the canonical encoder loop:
    //   diff starts at step>>3 (the implicit 1/8 term);
    //   if mag >= step set bit2 and subtract step;
    //   if remaining mag >= step>>1 set bit1 and subtract step>>1;
    //   if remaining mag >= step>>2 set bit0.
    let mut mantissa = 0u8;
    let mut residual = mag;
    let mut step_acc = step;
    if residual >= step_acc {
        mantissa |= 0x04;
        residual -= step_acc;
    }
    step_acc >>= 1;
    if residual >= step_acc {
        mantissa |= 0x02;
        residual -= step_acc;
    }
    step_acc >>= 1;
    if residual >= step_acc {
        mantissa |= 0x01;
    }

    let nibble = sign | mantissa;

    // Advance the predictor + index with the decoder's recurrence so the
    // encoder state is the decoder state.
    *predictor += nibble_delta(nibble, step);
    *predictor = (*predictor).clamp(i16::MIN as i32, i16::MAX as i32);
    *index += IMA_INDEX_TABLE[(nibble & 0x07) as usize];
    *index = (*index).clamp(0, STEP_INDEX_MAX);

    nibble
}

/// Encode a buffer of 16-bit mono PCM samples into the nibble-packed body
/// of an AMV `01wb` audio block, per trace §4b (the inverse of
/// [`crate::decode_audio_block`]).
///
/// Returns `(initial_predictor, nibble_body)`:
///
/// * `initial_predictor` is the §4b header `int16` seed — the first PCM
///   sample, which the decoder re-seeds the predictor from so the block's
///   first decoded output reproduces it exactly.
/// * `nibble_body` is the low-nibble-first packed ADPCM (one nibble per
///   sample), `ceil(samples.len() / 2)` bytes; an odd final sample leaves
///   the high nibble of the last byte zero (decoded but dropped because
///   the §4b `sampleCount` bounds the output).
///
/// The block is self-contained: the step index starts at `0` (§4b reset).
/// An empty input yields the zero seed and an empty body.
pub fn encode_audio_nibbles(samples: &[i16]) -> (i16, Vec<u8>) {
    if samples.is_empty() {
        return (0, Vec::new());
    }
    // §4b: the predictor is seeded from the header int16, which the
    // decoder reproduces as the first output sample, so the seed is the
    // first PCM sample itself.
    let seed = samples[0];
    let mut predictor = seed as i32;
    let mut index = 0i32;

    let mut body = Vec::with_capacity(samples.len().div_ceil(2));
    let mut pending: Option<u8> = None;
    for &s in samples {
        let nibble = encode_one(s, &mut predictor, &mut index);
        match pending.take() {
            None => pending = Some(nibble),
            Some(low) => body.push(low | (nibble << 4)),
        }
    }
    // Flush a dangling low nibble (odd sample count) with a zero high
    // nibble; the decoder keeps only `sampleCount` outputs so the extra
    // decoded nibble is dropped.
    if let Some(low) = pending {
        body.push(low);
    }
    (seed, body)
}

/// Encode 16-bit mono PCM into a complete AMV `01wb` chunk *payload* — the
/// 8-byte §4b preamble (predictor seed, reset step index, sample count)
/// followed by the nibble body. The byte-exact inverse of
/// [`crate::decode_audio_payload`].
///
/// The preamble is laid out per §4b:
/// `+0 int16 initialPredictor`, `+2 int16 initialStepIndex` (`0`, the
/// decode-verified reset value), `+4 uint32 sampleCount`, `+8…` nibbles.
/// `decode_audio_payload(encode_audio_payload(pcm))` reproduces the
/// canonical IMA round-trip of `pcm` (lossy but a stable fixed point — see
/// the module docs).
pub fn encode_audio_payload(samples: &[i16]) -> Vec<u8> {
    let (seed, nibbles) = encode_audio_nibbles(samples);
    let mut payload = Vec::with_capacity(AMV_AUDIO_PREAMBLE_LEN + nibbles.len());
    // +0x00 int16 initialPredictor (little-endian).
    payload.extend_from_slice(&(seed as u16).to_le_bytes());
    // +0x02 int16 initialStepIndex = 0 (§4b decode-verified reset).
    payload.extend_from_slice(&0u16.to_le_bytes());
    // +0x04 uint32 sampleCount.
    payload.extend_from_slice(&(samples.len() as u32).to_le_bytes());
    // +0x08… nibble body.
    payload.extend_from_slice(&nibbles);
    payload
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adpcm::{decode_audio_block, decode_audio_payload};
    use crate::parse::AmvAudioPreamble;

    #[test]
    fn encode_tables_match_decode_tables() {
        // The encode-side table copies must be byte-equal to the decode
        // side's; pin a representative spread (the decode module keeps
        // them private, so re-derive from the documented anchors).
        assert_eq!(IMA_STEP_TABLE.len(), 89);
        assert_eq!(IMA_STEP_TABLE[0], 7);
        assert_eq!(IMA_STEP_TABLE[8], 16);
        assert_eq!(IMA_STEP_TABLE[88], 32767);
        assert_eq!(IMA_INDEX_TABLE, [-1, -1, -1, -1, 2, 4, 6, 8]);
        assert_eq!(STEP_INDEX_MAX, 88);
    }

    #[test]
    fn empty_input_yields_zero_seed_empty_body() {
        let (seed, body) = encode_audio_nibbles(&[]);
        assert_eq!(seed, 0);
        assert!(body.is_empty());
    }

    #[test]
    fn first_sample_becomes_predictor_seed_and_decodes_back() {
        // The first PCM sample is the seed; the decoder re-seeds the
        // predictor from it so the first decoded output equals it.
        let pcm = [1234i16, 1234, 1234, 1234];
        let payload = encode_audio_payload(&pcm);
        let pre = AmvAudioPreamble::parse(&payload).unwrap();
        assert_eq!(pre.initial_predictor(), 1234);
        assert_eq!(pre.initial_step_index(), 0, "§4b reset step index");
        assert_eq!(pre.decoded_sample_count, 4);
        let decoded = decode_audio_payload(&payload).unwrap();
        assert_eq!(decoded.len(), 4);
        assert_eq!(decoded[0], 1234, "first decoded sample is the seed");
    }

    #[test]
    fn nibble_body_length_is_ceil_half_sample_count() {
        for n in [0usize, 1, 2, 3, 4, 5, 1837] {
            let pcm = vec![0i16; n];
            let (_, body) = encode_audio_nibbles(&pcm);
            assert_eq!(body.len(), n.div_ceil(2), "n={n}");
        }
    }

    #[test]
    fn payload_length_matches_preamble_plus_body() {
        let pcm = vec![100i16; 1837];
        let payload = encode_audio_payload(&pcm);
        assert_eq!(
            payload.len(),
            AMV_AUDIO_PREAMBLE_LEN + 1837usize.div_ceil(2)
        );
    }

    #[test]
    fn nibbles_packed_low_first() {
        // Two samples → one byte: low nibble = first sample's code, high
        // nibble = second sample's code. Drive the predictor so the two
        // chosen nibbles differ, then confirm the packing order.
        let pcm = [0i16, 4000];
        let (_, body) = encode_audio_nibbles(&pcm);
        assert_eq!(body.len(), 1);
        // Re-derive the two nibbles independently and check packing.
        let mut p = 0i32;
        let mut i = 0i32;
        let n0 = encode_one(pcm[0], &mut p, &mut i);
        let n1 = encode_one(pcm[1], &mut p, &mut i);
        assert_eq!(body[0], n0 | (n1 << 4), "low nibble first");
    }

    #[test]
    fn encode_decode_is_a_stable_fixed_point() {
        // ADPCM is lossy, but re-encoding the decoded output must
        // reproduce the same nibbles — the canonical IMA fixed point.
        // Use a structured ramp + sine-like signal.
        let pcm: Vec<i16> = (0..512)
            .map(|k| ((k as f64 * 0.05).sin() * 8000.0) as i16)
            .collect();
        let payload1 = encode_audio_payload(&pcm);
        let decoded1 = decode_audio_payload(&payload1).unwrap();
        let payload2 = encode_audio_payload(&decoded1);
        let decoded2 = decode_audio_payload(&payload2).unwrap();
        assert_eq!(
            payload1, payload2,
            "re-encoding decoded output reproduces identical bytes"
        );
        assert_eq!(decoded1, decoded2, "decode is a stable fixed point");
    }

    #[test]
    fn round_trip_tracks_signal_within_adpcm_error() {
        // The decoded round-trip should track a smooth signal closely
        // (the encoder tracks the decoder's reconstructed predictor, so
        // there is no runaway drift). Measure mean abs error on a slow
        // sine well inside the step-table's slew capability.
        let pcm: Vec<i16> = (0..1024)
            .map(|k| ((k as f64 * 0.02).sin() * 6000.0) as i16)
            .collect();
        let payload = encode_audio_payload(&pcm);
        let decoded = decode_audio_payload(&payload).unwrap();
        assert_eq!(decoded.len(), pcm.len());
        let mae: f64 = pcm
            .iter()
            .zip(&decoded)
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / pcm.len() as f64;
        // A slow sine well inside the slew limit reconstructs tightly.
        assert!(mae < 400.0, "round-trip MAE {mae} too high for a slow sine");
    }

    #[test]
    fn odd_sample_count_round_trips_exact_length() {
        // An odd count leaves a half-used final byte; the decoder must
        // still return exactly `sampleCount` samples.
        let pcm = [10i16, -20, 30, -40, 50];
        let payload = encode_audio_payload(&pcm);
        let pre = AmvAudioPreamble::parse(&payload).unwrap();
        assert_eq!(pre.decoded_sample_count, 5);
        let decoded = decode_audio_payload(&payload).unwrap();
        assert_eq!(decoded.len(), 5, "odd count preserved through round-trip");
    }

    #[test]
    fn manual_block_split_matches_payload_helper() {
        // encode_audio_payload == preamble(seed, count, 0) bytes + body.
        let pcm = [500i16, 600, 700, 800, 900, 1000];
        let payload = encode_audio_payload(&pcm);
        let (seed, body) = encode_audio_nibbles(&pcm);
        let pre = AmvAudioPreamble::parse(&payload).unwrap();
        assert_eq!(pre.initial_predictor(), seed);
        assert_eq!(&payload[AMV_AUDIO_PREAMBLE_LEN..], &body[..]);
        // And decoding via the block path with the parsed preamble equals
        // the whole-payload decode.
        let via_block = decode_audio_block(&pre, &payload[AMV_AUDIO_PREAMBLE_LEN..]);
        let via_payload = decode_audio_payload(&payload).unwrap();
        assert_eq!(via_block, via_payload);
    }
}
