//! Nibble-to-PCM decode for an AMV `01wb` audio block — the §4b
//! "IMA step / index tables — STANDARD" step of
//! `docs/container/amv/amv-container-trace.md`.
//!
//! # Why this lives in the container crate
//!
//! This is the audio counterpart of [`crate::reconstruct_jpeg`]. Trace
//! §4b establishes that an AMV `01wb` block is **standard** IMA/DVI
//! ADPCM — the 89-entry step-size table and the 8-entry index-adjust
//! table are the canonical IMA tables, used unmodified; only the
//! *container framing* is AMV-specific (the 8-byte per-block header with
//! an `int16` predictor seed + reset step index, one block per video
//! frame, low-nibble-first packing). Applying those public tables to the
//! block's nibbles to produce the 16-bit PCM the §3b `WAVEFORMATEX`
//! *declares* is the byte-exact realisation of the AMV block: the same
//! decode-adjacent wire glue the JPEG marker-splice is on the video
//! side. No DSP transform of the device's own design happens here — the
//! recurrence is the public IMA/DVI ADPCM recurrence, and the only thing
//! AMV contributes is *where* the predictor seed and sample count come
//! from (the §4b header).
//!
//! # What the device hardcodes (trace §4b, decode-verified)
//!
//! Per trace §4b, decoding all 1116 blocks of the staged `comedian.amv`
//! with these standard tables — seeding each block's predictor from its
//! header `int16` and resetting the step index to `0` — yields
//! 2 050 650 mono samples = exactly 93.0 s at 22 050 Hz, matching the
//! §2 container duration (1:33), with only 0.024 % of samples hitting
//! the ±32768 clamp and a speech-like low-pass spectrum. Wrong tables,
//! carrying predictor state across blocks, or treating header `+0x02`
//! as the step index all produce heavy clamping / noise — i.e. the
//! standard-tables finding is established by the decoded audio being
//! sane, not by reading any decoder.
//!
//! No ADPCM **decoder** source was read; the standard IMA step / index
//! tables are public (the IMA/DVI ADPCM recommendation) and are
//! reproduced here as `const` arrays because the `docs/` submodule is
//! not part of the published crate.

use crate::parse::AmvAudioPreamble;

// ---------------------------------------------------------------------
// Standard IMA / DVI ADPCM decode tables (trace §4b — used verbatim,
// not an AMV variant). Public IMA/DVI definition.
// ---------------------------------------------------------------------

/// Canonical IMA/DVI 89-entry step-size table (trace §4b: "7, 8, 9, 10,
/// 11, 12, 13, 14, 16, … 27086, 29794, 32767"). Indexed by the running
/// step index, which is clamped to `[0, 88]`.
const IMA_STEP_TABLE: [i32; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449,
    494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272,
    2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
    10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
];

/// Canonical IMA/DVI 8-entry index-adjust table (trace §4b:
/// `{ -1, -1, -1, -1, 2, 4, 6, 8 }`), indexed by the low 3 bits of the
/// 4-bit nibble.
const IMA_INDEX_TABLE: [i32; 8] = [-1, -1, -1, -1, 2, 4, 6, 8];

/// Maximum valid step index — the 89-entry [`IMA_STEP_TABLE`] is indexed
/// `[0, 88]`. (Mirrors the `i16`-typed [`crate::IMA_STEP_INDEX_MAX`]
/// preamble-range bound, here as the table's `usize` upper index.)
const STEP_INDEX_MAX: i32 = (IMA_STEP_TABLE.len() - 1) as i32;

/// Decode one 4-bit ADPCM nibble against a running `(predictor, index)`
/// state, returning the new clamped 16-bit predictor. The recurrence is
/// the canonical IMA one the trace §4b records verbatim:
///
/// > `diff = step>>3`; `+= step>>2` if bit0; `+= step>>1` if bit1;
/// > `+= step` if bit2; negate if bit3 (sign); predictor clamped to
/// > int16; index clamped to `[0, 88]`.
#[inline]
fn decode_nibble(nibble: u8, predictor: &mut i32, index: &mut i32) -> i16 {
    let step = IMA_STEP_TABLE[*index as usize];

    // diff = step/8 + (bit2 ? step : 0) + (bit1 ? step/2 : 0) + (bit0 ? step/4 : 0)
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
    // bit3 is the sign.
    if nibble & 0x08 != 0 {
        *predictor -= diff;
    } else {
        *predictor += diff;
    }
    *predictor = (*predictor).clamp(i16::MIN as i32, i16::MAX as i32);

    // Advance and clamp the step index by the low 3 bits of the nibble.
    *index += IMA_INDEX_TABLE[(nibble & 0x07) as usize];
    *index = (*index).clamp(0, STEP_INDEX_MAX);

    *predictor as i16
}

/// Decode the compressed body of an AMV `01wb` audio block into 16-bit
/// PCM mono samples, per trace §4b.
///
/// `preamble` is the parsed 8-byte block header (its
/// [`AmvAudioPreamble::initial_predictor`] seeds the predictor and
/// [`AmvAudioPreamble::decoded_sample_count`] bounds the output);
/// `compressed_body` is the nibble-packed payload *after* the 8-byte
/// preamble (i.e. `&01wb_payload[AMV_AUDIO_PREAMBLE_LEN..]`).
///
/// Trace §4b decode contract, applied exactly:
///
/// * The block is **self-contained** — the predictor is re-seeded from
///   the header `int16` ([`AmvAudioPreamble::initial_predictor`]) and
///   the step index is **reset to 0** at the start of every block (no
///   state carries across blocks). The step index is *not* taken from
///   the preamble `+0x02` field. The trace's §4b "refined header layout"
///   reports that field as "always `00 00`", but it is in fact non-zero
///   in some blocks of the staged fixture (e.g. `30` at audio block 50);
///   the §4b gap note records that "treating header +2 as the step index
///   made the output worse", and only an index-0 reset reproduces the
///   decode-verified sanity numbers below. The validated decode therefore
///   ignores `+0x02`.
/// * Nibbles are unpacked **low-nibble-first** (the standard IMA byte
///   order): byte `b` yields nibble `b & 0x0F` then `b >> 4`.
/// * Exactly `decoded_sample_count` outputs are kept (the trace's
///   "919 body bytes give 1838 nibbles, and the first `sampleCount`
///   decoded outputs are kept"); if the body holds fewer nibbles than
///   the declared count the decode stops at the body end (truncation
///   tolerance, matching the demuxer's §4c behaviour).
///
/// The result is the 16-bit PCM the §3b `WAVEFORMATEX` declares. No
/// device-specific DSP is performed — the recurrence is the public IMA
/// one (see [`decode_nibble`]).
pub fn decode_audio_block(preamble: &AmvAudioPreamble, compressed_body: &[u8]) -> Vec<i16> {
    let want = preamble.decoded_sample_count as usize;
    let mut out = Vec::with_capacity(want);

    // §4b: per-block re-seed of the predictor from the header `int16`,
    // and the step index is **reset to 0** at block start — the trace's
    // decode-verified recipe ("seeding each block's predictor from its
    // header int16 and resetting the step index to 0"). The step index is
    // *not* seeded from the preamble `+0x02` field: the trace's §4b gap
    // note records that "treating header +2 as the step index made the
    // output worse", and decoding with index 0 is what reproduces the
    // 0.024 % clip rate / 93.0 s sanity result. (The on-disk `+0x02`
    // field is not in fact always zero across all blocks — see
    // `decode_audio_block` doc — but the validated decode ignores it.)
    let mut predictor = preamble.initial_predictor() as i32;
    let mut index = 0i32;

    'outer: for &byte in compressed_body {
        // Low nibble first, then high nibble.
        for nibble in [byte & 0x0F, byte >> 4] {
            if out.len() == want {
                break 'outer;
            }
            out.push(decode_nibble(nibble, &mut predictor, &mut index));
        }
    }
    out
}

/// Decode a whole `01wb` audio-chunk *payload* (the §4b 8-byte preamble
/// **plus** its compressed body) into 16-bit PCM mono samples — the audio
/// counterpart of [`crate::reconstruct_jpeg_from_payload`], which takes a
/// raw `00dc` payload.
///
/// `payload` is the full leaf-chunk body a [`crate::MoviPayload::Audio`]
/// hands back (or `&01wb_chunk_bytes`): this parses the §4b preamble off
/// the front via [`AmvAudioPreamble::parse`] (seeding the predictor from
/// `+0x00` and the sample count from `+0x04`), then decodes the remaining
/// bytes with [`decode_audio_block`]. It removes the manual
/// `AmvAudioPreamble::parse` + `&payload[AMV_AUDIO_PREAMBLE_LEN..]` slice
/// step every consumer would otherwise repeat.
///
/// Returns the demuxer error if `payload` is shorter than the 8-byte
/// preamble (`AMV_AUDIO_PREAMBLE_LEN`). A payload that is exactly the
/// preamble with no compressed body decodes to an empty `Vec` (the §4b
/// degenerate empty block).
pub fn decode_audio_payload(payload: &[u8]) -> Result<Vec<i16>, crate::AmvDemuxerError> {
    let preamble = AmvAudioPreamble::parse(payload)?;
    Ok(decode_audio_block(
        &preamble,
        &payload[crate::parse::AMV_AUDIO_PREAMBLE_LEN..],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::AmvHeader;

    /// A preamble with a given predictor seed, step index 0, and sample
    /// count, mirroring the on-disk §4b layout (state low 16 bits =
    /// predictor, high 16 bits = step index).
    fn preamble(predictor: i16, count: u32) -> AmvAudioPreamble {
        AmvAudioPreamble {
            state: (predictor as u16) as u32, // step index 0 in high half
            decoded_sample_count: count,
        }
    }

    #[test]
    fn step_and_index_tables_are_canonical_ima() {
        // Trace §4b: 89-entry step table bounded by 7 … 32767.
        assert_eq!(IMA_STEP_TABLE.len(), 89);
        assert_eq!(IMA_STEP_TABLE[0], 7);
        assert_eq!(IMA_STEP_TABLE[88], 32767);
        // A few interior anchors the trace names explicitly.
        assert_eq!(IMA_STEP_TABLE[8], 16);
        assert_eq!(IMA_STEP_TABLE[86], 27086);
        assert_eq!(IMA_STEP_TABLE[87], 29794);
        // 8-entry index-adjust table.
        assert_eq!(IMA_INDEX_TABLE, [-1, -1, -1, -1, 2, 4, 6, 8]);
        assert_eq!(STEP_INDEX_MAX, 88);
    }

    #[test]
    fn nibble_recurrence_matches_trace_formula() {
        // From predictor 0, step index 0 (step = 7):
        // nibble 0b0100 (bit2 set, sign clear):
        //   diff = 7>>3 (=0) + 7 = 7; predictor 0 + 7 = 7.
        //   index += IMA_INDEX_TABLE[4] = +2 -> 2.
        let mut p = 0i32;
        let mut i = 0i32;
        let s = decode_nibble(0b0100, &mut p, &mut i);
        assert_eq!(s, 7);
        assert_eq!(p, 7);
        assert_eq!(i, 2);

        // nibble 0b1100 (bit2 + sign): step now 9 (index 2).
        //   diff = 9>>3 (=1) + 9 = 10; predictor 7 - 10 = -3 (sign set).
        //   index += IMA_INDEX_TABLE[4] = +2 -> 4.
        let s = decode_nibble(0b1100, &mut p, &mut i);
        assert_eq!(s, -3);
        assert_eq!(p, -3);
        assert_eq!(i, 4);
    }

    #[test]
    fn nibble_zero_does_not_move_index_below_zero() {
        // nibble 0 -> index += IMA_INDEX_TABLE[0] = -1, clamped to 0.
        let mut p = 0i32;
        let mut i = 0i32;
        // diff = step>>3 = 7>>3 = 0; predictor unchanged.
        let s = decode_nibble(0, &mut p, &mut i);
        assert_eq!(s, 0);
        assert_eq!(p, 0);
        assert_eq!(i, 0, "index clamps at 0, never negative");
    }

    #[test]
    fn predictor_clamps_to_int16_range() {
        // Force the predictor toward the positive rail with repeated
        // max-magnitude positive nibbles at the top step.
        let mut p = i16::MAX as i32 - 10;
        let mut i = STEP_INDEX_MAX;
        // nibble 0b0111: positive, diff = step>>3 + step + step>>1 + step>>2,
        // a large jump that must clamp.
        let s = decode_nibble(0b0111, &mut p, &mut i);
        assert_eq!(s, i16::MAX);
        assert_eq!(p, i16::MAX as i32, "clamped to +32767");
        // The index advanced but clamps at 88.
        assert_eq!(i, STEP_INDEX_MAX);
    }

    #[test]
    fn step_index_clamps_at_max() {
        // Drive the index up with high nibbles (index += 8) and confirm
        // it never exceeds 88, which would index out of bounds.
        let mut p = 0i32;
        let mut i = 80i32;
        for _ in 0..20 {
            // nibble 0b0111: low 3 bits = 7 -> IMA_INDEX_TABLE[7] = +8.
            let _ = decode_nibble(0b0111, &mut p, &mut i);
        }
        assert_eq!(i, STEP_INDEX_MAX, "step index saturates at 88");
    }

    #[test]
    fn low_nibble_decoded_before_high_nibble() {
        // One byte 0x40: low nibble = 0, high nibble = 4.
        // Low nibble 0 first (no predictor change), then nibble 4
        // (diff = 7>>3 + 7 = 7).
        let pre = preamble(0, 2);
        let out = decode_audio_block(&pre, &[0x40]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], 0, "low nibble (0) decoded first");
        assert_eq!(out[1], 7, "high nibble (4) decoded second");
    }

    #[test]
    fn output_truncated_to_decoded_sample_count() {
        // Two body bytes carry 4 nibbles, but the header asks for 3.
        let pre = preamble(0, 3);
        let out = decode_audio_block(&pre, &[0x44, 0x44]);
        assert_eq!(out.len(), 3, "kept exactly decoded_sample_count outputs");
    }

    #[test]
    fn output_stops_at_body_end_when_short() {
        // Header declares 100 samples but the body only has one byte
        // (2 nibbles): decode stops at the body end (truncation).
        let pre = preamble(0, 100);
        let out = decode_audio_block(&pre, &[0x44]);
        assert_eq!(out.len(), 2, "decode bounded by available nibbles");
    }

    #[test]
    fn empty_body_yields_no_samples() {
        let pre = preamble(0, 10);
        let out = decode_audio_block(&pre, &[]);
        assert!(out.is_empty());
    }

    #[test]
    fn block_reseeds_predictor_from_header() {
        // A non-zero predictor seed (§4b: blocks 0-7 are 0,1,-9,8,…).
        // With nibble 0 (no change) the first output equals the seed.
        let pre = preamble(-9, 1);
        let out = decode_audio_block(&pre, &[0x00]);
        assert_eq!(out[0], -9, "first sample reflects re-seeded predictor");
    }

    #[test]
    fn step_index_resets_to_zero_ignoring_preamble_plus2() {
        // The validated decode resets the step index to 0 regardless of
        // the preamble `+0x02` field (which is *not* always zero on disk —
        // e.g. 30 at fixture audio block 50). Build a preamble whose
        // high-half (the `+0x02` int16) is non-zero and confirm the first
        // nibble decodes at step index 0 (step = 7), not at the preamble's
        // value.
        let pre = AmvAudioPreamble {
            // low 16 bits = predictor 0; high 16 bits = step index 30.
            state: (30u32 << 16),
            decoded_sample_count: 1,
        };
        assert_eq!(pre.initial_step_index(), 30, "preamble +0x02 surfaces 30");
        // nibble 0b0100 at step index 0: diff = 7>>3 + 7 = 7 -> output 7.
        // If the decode had (wrongly) seeded index 30 (step = 130),
        // the output would be 130>>3 + 130 = 146.
        let out = decode_audio_block(&pre, &[0x04]);
        assert_eq!(out[0], 7, "decode resets step index to 0, ignoring +0x02");
    }

    #[test]
    fn decode_audio_payload_matches_manual_preamble_split() {
        // The whole-payload convenience equals parse-preamble + decode-body.
        // Build a payload: predictor seed -9 (low 16 of state), step index 0,
        // sample count 4, then two body bytes (4 nibbles).
        let mut payload = Vec::new();
        payload.extend_from_slice(&((-9i16 as u16) as u32).to_le_bytes()); // state +0x00
        payload.extend_from_slice(&4u32.to_le_bytes()); // count +0x04
        payload.extend_from_slice(&[0x40, 0x44]); // body
        let via_payload = decode_audio_payload(&payload).expect("payload decodes");
        let pre = AmvAudioPreamble::parse(&payload).unwrap();
        let via_block = decode_audio_block(&pre, &payload[8..]);
        assert_eq!(via_payload, via_block);
        assert_eq!(via_payload.len(), 4);
        assert_eq!(
            via_payload[0], -9,
            "first sample is the re-seeded predictor"
        );
    }

    #[test]
    fn decode_audio_payload_preamble_only_is_empty() {
        // Exactly 8 preamble bytes, no body -> empty PCM.
        let mut payload = Vec::new();
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&10u32.to_le_bytes());
        let out = decode_audio_payload(&payload).expect("preamble-only decodes");
        assert!(out.is_empty(), "no body yields no samples");
    }

    #[test]
    fn decode_audio_payload_rejects_short_preamble() {
        // Fewer than 8 bytes cannot carry a preamble.
        assert!(decode_audio_payload(&[0x00, 0x01, 0x02]).is_err());
    }

    fn comedian_header() -> AmvHeader {
        AmvHeader {
            micros_per_frame: 83_333,
            width: 128,
            height: 96,
            fps: 12,
            flag_one: 1,
            reserved_30: 0,
            duration_packed: 0x0000_0121,
        }
    }

    /// Real-fixture decode pin: decode every `01wb` block of
    /// `comedian.amv` with the standard IMA tables and confirm the §4b
    /// decode-sanity numbers — 2 050 650 mono samples = exactly 93.0 s
    /// at 22 050 Hz, with a sub-0.1 % ±32768 clip rate (trace §4b
    /// records 0.024 %).
    #[test]
    fn comedian_fixture_decodes_to_93s_low_clip() {
        use crate::parse::{
            AmvAudioPreamble, MoviPayload, MoviPayloadIter, AMV_AUDIO_PREAMBLE_LEN, AMV_END_TRAILER,
        };

        let crate_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/comedian.amv");
        let workspace_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/container/amv/fixtures/comedian.amv");
        let path = if crate_path.exists() {
            crate_path
        } else if workspace_path.exists() {
            workspace_path
        } else {
            eprintln!("skipping comedian adpcm fixture test: not staged");
            return;
        };
        let bytes = std::fs::read(&path).expect("read fixture");
        let _ = comedian_header(); // header geometry not needed for audio.

        let movi_pos = bytes.windows(4).position(|w| w == b"movi").unwrap();
        let trailer_start = bytes.len() - AMV_END_TRAILER.len();
        let movi_body = &bytes[movi_pos + 4..trailer_start];

        let mut total_samples: u64 = 0;
        let mut clipped: u64 = 0;
        let mut blocks = 0u32;
        let mut first_block_len = 0usize;

        for payload in MoviPayloadIter::new(movi_body).filter_map(|r| r.ok()) {
            if let MoviPayload::Audio { body, .. } = payload {
                // Whole-payload convenience: equals the manual
                // parse-preamble + decode-body split (cross-checked on the
                // first block).
                let pcm = decode_audio_payload(body).expect("01wb payload decodes");
                if blocks == 0 {
                    let preamble = AmvAudioPreamble::parse(body).expect("preamble");
                    let split = decode_audio_block(&preamble, &body[AMV_AUDIO_PREAMBLE_LEN..]);
                    assert_eq!(pcm, split, "payload convenience == manual split");
                    first_block_len = pcm.len();
                }
                for &s in &pcm {
                    if s == i16::MIN || s == i16::MAX {
                        clipped += 1;
                    }
                }
                total_samples += pcm.len() as u64;
                blocks += 1;
            }
        }

        assert_eq!(blocks, 1116, "§4 1116 audio blocks");
        assert_eq!(
            first_block_len, 1837,
            "§4b first block decodes 1837 samples"
        );
        // §4b: 2 050 650 mono samples = exactly 93.0 s @ 22 050 Hz.
        assert_eq!(total_samples, 2_050_650, "§4b total decoded sample count");
        assert_eq!(total_samples / 22_050, 93, "exactly 93.0 s at 22 050 Hz");
        // §4b records a 0.024 % clip rate; assert a generous sub-0.1 %
        // bound so the standard-tables finding holds (wrong tables ran
        // at 0.6 %+).
        let clip_rate = clipped as f64 / total_samples as f64;
        assert!(
            clip_rate < 0.001,
            "clip rate {clip_rate} should be well under 0.1% (trace §4b: 0.024%)"
        );
    }
}
