//! Reassembly of the device-stripped JPEG marker segments for an AMV
//! `00dc` video frame — the §4a "reconstruction" step of
//! `docs/container/amv/amv-container-trace.md`.
//!
//! # Why this lives in the container crate
//!
//! An AMV `00dc` payload is a JPEG bitstream with its marker segments
//! **removed**: per trace §4a a frame is `FF D8` (SOI) … entropy-coded
//! data … `FF D9` (EOI) with *no* `DQT` / `SOF0` / `DHT` / `SOS` in
//! between. "The quantization and Huffman tables, frame geometry and
//! scan parameters are all stripped and hardcoded in the player's
//! decoder; the resolution comes from `amvh`." The device's encoder
//! deletes header bytes that a conforming reader requires, and the
//! player splices fixed bytes back in before decode.
//!
//! Putting those bytes back is **wire-format reconstruction**, not
//! image decoding: it is the byte-exact inverse of the container's
//! header-stripping, and it is the AMV-specific glue without which a
//! generic baseline-JPEG decoder cannot consume a frame. No DCT, no
//! Huffman walk, no dequantisation happens here — the entropy-coded
//! bytes are copied through untouched. The output is a standards-
//! conforming baseline JFIF/JPEG that any MJPEG/JPEG decoder
//! (`oxideav-mjpeg` downstream) accepts unchanged. This mirrors the
//! `lib.rs` note that "the AMV player splices its tables back in before
//! handing the bitstream to a standard JPEG decoder".
//!
//! # What the device hardcodes (trace §4a, reconstruction-verified)
//!
//! Trace §4a established by reconstruction-and-decode that the stripped
//! segments are, verbatim and unscaled:
//!
//! * **Quantization** — JPEG Annex K example tables K.1 (luma, `Tq=0`)
//!   and K.2 (chroma, `Tq=1`), 8-bit precision, written in zig-zag
//!   order per T.81 §B.2.4.1.
//! * **Frame** — baseline sequential DCT (`SOF0`), 8-bit precision,
//!   3 components Y/Cb/Cr, **4:2:0** sampling (luma `Hi=2 Vi=2`, chroma
//!   `1×1`), dimensions from the §2 `amvh` header.
//! * **Huffman** — JPEG Annex K example tables K.3 (luma DC+AC) and K.4
//!   (chroma DC+AC), used verbatim, in one `DHT` segment.
//! * **Scan** — single interleaved `SOS` over all three components,
//!   natural spectral selection `Ss=0 Se=63 Ah=0 Al=0`.
//!
//! All table values below are the public JPEG Annex K (ITU-T T.81)
//! example tables, transcribed from the clean-room extracts in
//! `docs/image/jpeg/tables/` (which the trace §4a cites by name); they
//! are reproduced as `const` arrays here because the `docs/` submodule
//! is not part of the published crate. No JPEG/AMV **decoder** source
//! was read — only the public table data and the public T.81 marker
//! syntax (Annex B).

use crate::parse::AmvHeader;
use crate::video::AmvVideoFrame;

// ---------------------------------------------------------------------
// JPEG Annex K example tables (ITU-T T.81), natural / row-major order.
// Source: docs/image/jpeg/tables/*.csv (clean-room data extracts, cited
// by trace §4a). Reproduced as constants — see module docs.
// ---------------------------------------------------------------------

/// Annex K Table K.1 — default **luminance** quantization table, in
/// natural (row-major) 8×8 order. `Tq=0`. (docs `…/default-quant-table-luminance.csv`.)
pub(crate) const QUANT_LUMA: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, //
    12, 12, 14, 19, 26, 58, 60, 55, //
    14, 13, 16, 24, 40, 57, 69, 56, //
    14, 17, 22, 29, 51, 87, 80, 62, //
    18, 22, 37, 56, 68, 109, 103, 77, //
    24, 35, 55, 64, 81, 104, 113, 92, //
    49, 64, 78, 87, 103, 121, 120, 101, //
    72, 92, 95, 98, 112, 100, 103, 99,
];

/// Annex K Table K.2 — default **chrominance** quantization table, in
/// natural (row-major) 8×8 order. `Tq=1`. (docs `…/default-quant-table-chrominance.csv`.)
pub(crate) const QUANT_CHROMA: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, //
    18, 21, 26, 66, 99, 99, 99, 99, //
    24, 26, 56, 99, 99, 99, 99, 99, //
    47, 66, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99, //
    99, 99, 99, 99, 99, 99, 99, 99,
];

/// Zig-zag scan order (T.81 §A.1.4 / §B.2.4.1): for each *natural*
/// 8×8 position, the index in transmission (zig-zag) order. A DQT
/// segment transmits the table in zig-zag order, so element `k` of the
/// emitted byte run is the natural-order value whose zig-zag index is
/// `k`. (docs `…/zigzag-scan-order.csv`.)
pub(crate) const ZIGZAG: [u8; 64] = [
    0, 1, 5, 6, 14, 15, 27, 28, //
    2, 4, 7, 13, 16, 26, 29, 42, //
    3, 8, 12, 17, 25, 30, 41, 43, //
    9, 11, 18, 24, 31, 40, 44, 53, //
    10, 19, 23, 32, 39, 45, 52, 54, //
    20, 22, 33, 38, 46, 51, 55, 60, //
    21, 34, 37, 47, 50, 56, 59, 61, //
    35, 36, 48, 49, 57, 58, 62, 63,
];

// Huffman BITS arrays are the 16 counts of codes per length 1..=16
// (the docs CSVs carry a leading 0 at index 0 per the libjpeg
// 17-entry convention; only entries 1..=16 are transmitted in a DHT).

/// K.3 luma DC `BITS` (lengths 1..=16). (docs `…/huffman-dc-luminance-bits.csv`, indices 1..=16.)
pub(crate) const DC_LUMA_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
/// K.3 luma DC `HUFFVAL`. (docs `…/huffman-dc-luminance-val.csv`.)
pub(crate) const DC_LUMA_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// K.3 luma AC `BITS` (lengths 1..=16). (docs `…/huffman-ac-luminance-bits.csv`, indices 1..=16.)
pub(crate) const AC_LUMA_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
/// K.3 luma AC `HUFFVAL` (162 entries). (docs `…/huffman-ac-luminance-val.csv`.)
pub(crate) const AC_LUMA_VALS: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

/// K.4 chroma DC `BITS` (lengths 1..=16). (docs `…/huffman-dc-chrominance-bits.csv`, indices 1..=16.)
pub(crate) const DC_CHROMA_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
/// K.4 chroma DC `HUFFVAL`. (docs `…/huffman-dc-chrominance-val.csv`.)
pub(crate) const DC_CHROMA_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// K.4 chroma AC `BITS` (lengths 1..=16). (docs `…/huffman-ac-chrominance-bits.csv`, indices 1..=16.)
pub(crate) const AC_CHROMA_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
/// K.4 chroma AC `HUFFVAL` (162 entries). (docs `…/huffman-ac-chrominance-val.csv`.)
pub(crate) const AC_CHROMA_VALS: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

// JPEG marker bytes (T.81 Annex B). Only the segments the AMV device
// strips are emitted here; SOI / EOI already bracket the payload.
const MARKER_DQT: u8 = 0xDB;
const MARKER_SOF0: u8 = 0xC0;
const MARKER_DHT: u8 = 0xC4;
const MARKER_SOS: u8 = 0xDA;
const MARKER_PREFIX: u8 = 0xFF;

/// Push a big-endian 16-bit segment length (T.81: marker segment
/// lengths are 2 bytes, big-endian, and *include* the 2 length bytes).
fn push_be16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_be_bytes());
}

/// Append a `DQT` segment carrying both quantization tables (luma
/// `Tq=0`, chroma `Tq=1`), 8-bit precision, each transmitted in
/// zig-zag order (T.81 §B.2.4.1).
fn push_dqt(out: &mut Vec<u8>) {
    out.push(MARKER_PREFIX);
    out.push(MARKER_DQT);
    // Length: 2 (length field) + per table (1 Pq/Tq byte + 64 values).
    push_be16(out, 2 + 2 * (1 + 64));
    for (tq, table) in [(0u8, &QUANT_LUMA), (1u8, &QUANT_CHROMA)] {
        // Pq=0 (8-bit) in high nibble, Tq in low nibble.
        out.push(tq);
        for k in 0..64usize {
            // Emit in zig-zag order: the natural-order index whose
            // zig-zag position equals k.
            let natural = ZIGZAG.iter().position(|&z| z as usize == k).unwrap();
            out.push(table[natural]);
        }
    }
}

/// Append the baseline `SOF0` frame header for `width`×`height`,
/// 8-bit precision, 3 components, 4:2:0 (luma `2×2`, chroma `1×1`),
/// quant selectors Y→0, Cb/Cr→1.
fn push_sof0(out: &mut Vec<u8>, width: u16, height: u16) {
    out.push(MARKER_PREFIX);
    out.push(MARKER_SOF0);
    // Length: 8 fixed + 3 bytes per component × 3 components.
    push_be16(out, 8 + 3 * 3);
    out.push(8); // sample precision (P)
    push_be16(out, height); // Y (number of lines)
    push_be16(out, width); // X (samples per line)
    out.push(3); // Nf — component count
                 // Component 1: Y, H=2 V=2, quant table 0.
    out.extend_from_slice(&[1, 0x22, 0]);
    // Component 2: Cb, H=1 V=1, quant table 1.
    out.extend_from_slice(&[2, 0x11, 1]);
    // Component 3: Cr, H=1 V=1, quant table 1.
    out.extend_from_slice(&[3, 0x11, 1]);
}

/// Append one Huffman table to an in-progress `DHT` body.
/// `tc_th` packs class (Tc, high nibble) and destination (Th, low
/// nibble). T.81 §B.2.4.2: 1 Tc/Th byte, 16 BITS bytes, then HUFFVAL.
fn push_huff_table(out: &mut Vec<u8>, tc_th: u8, bits: &[u8; 16], vals: &[u8]) {
    out.push(tc_th);
    out.extend_from_slice(bits);
    out.extend_from_slice(vals);
}

/// Append a single `DHT` segment carrying all four Annex K tables:
/// luma DC (`Tc=0 Th=0`), luma AC (`Tc=1 Th=0`), chroma DC
/// (`Tc=0 Th=1`), chroma AC (`Tc=1 Th=1`).
fn push_dht(out: &mut Vec<u8>) {
    out.push(MARKER_PREFIX);
    out.push(MARKER_DHT);
    // Per-table byte cost = 1 (Tc/Th) + 16 (BITS) + HUFFVAL length.
    let len = 2
        + (1 + 16 + DC_LUMA_VALS.len())
        + (1 + 16 + AC_LUMA_VALS.len())
        + (1 + 16 + DC_CHROMA_VALS.len())
        + (1 + 16 + AC_CHROMA_VALS.len());
    push_be16(out, len as u16);
    push_huff_table(out, 0x00, &DC_LUMA_BITS, &DC_LUMA_VALS);
    push_huff_table(out, 0x10, &AC_LUMA_BITS, &AC_LUMA_VALS);
    push_huff_table(out, 0x01, &DC_CHROMA_BITS, &DC_CHROMA_VALS);
    push_huff_table(out, 0x11, &AC_CHROMA_BITS, &AC_CHROMA_VALS);
}

/// Append the single interleaved `SOS` header: 3 components, DC/AC
/// Huffman selectors Y→0/0, Cb→1/1, Cr→1/1, full spectral selection
/// `Ss=0 Se=63 Ah=0 Al=0`.
fn push_sos(out: &mut Vec<u8>) {
    out.push(MARKER_PREFIX);
    out.push(MARKER_SOS);
    // Length: 6 fixed + 2 bytes per component × 3.
    push_be16(out, 6 + 2 * 3);
    out.push(3); // Ns — component count in scan
                 // Cs1=Y: Td=0 (DC luma), Ta=0 (AC luma).
    out.extend_from_slice(&[1, 0x00]);
    // Cs2=Cb: Td=1 (DC chroma), Ta=1 (AC chroma).
    out.extend_from_slice(&[2, 0x11]);
    // Cs3=Cr: Td=1, Ta=1.
    out.extend_from_slice(&[3, 0x11]);
    out.push(0); // Ss
    out.push(63); // Se
    out.push(0); // Ah(0) | Al(0)
}

/// Reconstruct a conforming baseline JPEG from a bare AMV `00dc`
/// video frame.
///
/// Splices the device-stripped marker segments (§4a) into their
/// canonical T.81 order:
///
/// ```text
/// FF D8 (SOI, already in payload)
/// DQT   (K.1 luma Tq=0, K.2 chroma Tq=1, zig-zag order)
/// SOF0  (baseline, 8-bit, W×H, 4:2:0)
/// DHT   (K.3/K.4 luma+chroma DC+AC)
/// SOS   (single interleaved scan, Ss=0 Se=63)
/// <entropy-coded bytes — copied verbatim from the payload>
/// FF D9 (EOI, already in payload)
/// ```
///
/// The SOI and EOI are reused from the validated payload (the entropy
/// bytes are copied unchanged from [`AmvVideoFrame::entropy_coded`]);
/// the four header segments are inserted between SOI and the entropy
/// stream. The result is a complete baseline JFIF/JPEG that a generic
/// decoder accepts. No image decoding is performed.
pub fn reconstruct_jpeg(frame: &AmvVideoFrame<'_>) -> Vec<u8> {
    let entropy = frame.entropy_coded();
    let mut out = Vec::with_capacity(entropy.len() + 700);
    // SOI from the payload.
    out.extend_from_slice(&crate::parse::JPEG_SOI);
    push_dqt(&mut out);
    push_sof0(&mut out, frame.width() as u16, frame.height() as u16);
    push_dht(&mut out);
    push_sos(&mut out);
    // Entropy-coded scan data, byte-for-byte.
    out.extend_from_slice(entropy);
    // EOI from the payload.
    out.extend_from_slice(&crate::parse::JPEG_EOI);
    out
}

/// Convenience wrapper: bind the §2 `header` geometry to a raw `00dc`
/// `payload` (via [`AmvVideoFrame::bind_strict`], enforcing the §4a
/// no-internal-markers invariant) and reconstruct the conforming JPEG
/// in one step. Returns the demuxer error if the payload is not a
/// valid bare AMV frame.
pub fn reconstruct_jpeg_from_payload(
    header: &AmvHeader,
    payload: &[u8],
) -> Result<Vec<u8>, crate::AmvDemuxerError> {
    let frame = AmvVideoFrame::bind_strict(header, payload)?;
    Ok(reconstruct_jpeg(&frame))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::AMVH_BODY_LEN;

    /// §2 comedian device profile (128×96 @ 12 fps).
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

    /// A bare AMV frame: SOI + arbitrary entropy bytes + EOI.
    fn bare_frame(entropy: &[u8]) -> Vec<u8> {
        let mut v = vec![0xFF, 0xD8];
        v.extend_from_slice(entropy);
        v.extend_from_slice(&[0xFF, 0xD9]);
        v
    }

    /// Walk JPEG marker segments from offset 0, returning a list of
    /// `(marker_byte, payload_len_excluding_length_field)` until SOS
    /// (whose entropy data is not length-delimited) is reached.
    fn walk_markers(jpeg: &[u8]) -> Vec<(u8, usize)> {
        let mut out = Vec::new();
        let mut i = 0usize;
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "starts with SOI");
        i += 2;
        while i + 1 < jpeg.len() {
            assert_eq!(jpeg[i], 0xFF, "marker prefix at {i}");
            let m = jpeg[i + 1];
            i += 2;
            if m == 0xDA {
                // SOS: read its header length, record, then stop (rest
                // is entropy-coded scan data up to EOI).
                let len = u16::from_be_bytes([jpeg[i], jpeg[i + 1]]) as usize;
                out.push((m, len - 2));
                break;
            }
            let len = u16::from_be_bytes([jpeg[i], jpeg[i + 1]]) as usize;
            out.push((m, len - 2));
            i += len;
        }
        out
    }

    #[test]
    fn reconstructs_canonical_marker_sequence() {
        let frame_bytes = bare_frame(&[0xE6, 0x49, 0xA6, 0x93]);
        let h = comedian_header();
        let jpeg = reconstruct_jpeg_from_payload(&h, &frame_bytes).expect("binds + reconstructs");
        let markers = walk_markers(&jpeg);
        let tags: Vec<u8> = markers.iter().map(|&(m, _)| m).collect();
        // DQT, SOF0, DHT, SOS in canonical order.
        assert_eq!(tags, vec![MARKER_DQT, MARKER_SOF0, MARKER_DHT, MARKER_SOS]);
    }

    #[test]
    fn dqt_segment_length_and_zigzag_order() {
        let jpeg = reconstruct_jpeg(
            &AmvVideoFrame::bind(&comedian_header(), &bare_frame(&[0x11])).unwrap(),
        );
        // DQT begins right after SOI.
        assert_eq!(&jpeg[2..4], &[MARKER_PREFIX, MARKER_DQT]);
        let len = u16::from_be_bytes([jpeg[4], jpeg[5]]) as usize;
        // 2 (length) + 2 tables × (1 + 64).
        assert_eq!(len, 2 + 2 * 65);
        // First table: Pq/Tq byte = 0 (8-bit, luma Tq=0).
        assert_eq!(jpeg[6], 0x00);
        // First emitted quant value is the natural[0] (DC) = 16 for
        // luma — zig-zag index 0 maps to natural index 0.
        assert_eq!(jpeg[7], 16);
        // Second emitted value: zig-zag index 1 → natural index 1 = 11.
        assert_eq!(jpeg[8], 11);
        // Zig-zag index 2 → natural index 8 (start of row 1) = 12.
        assert_eq!(jpeg[9], 12);
        // Chroma table Pq/Tq byte follows the 64 luma values.
        assert_eq!(jpeg[6 + 65], 0x01);
        // Chroma DC value = 17.
        assert_eq!(jpeg[6 + 65 + 1], 17);
    }

    #[test]
    fn sof0_carries_geometry_and_420_sampling() {
        let jpeg = reconstruct_jpeg(
            &AmvVideoFrame::bind(&comedian_header(), &bare_frame(&[0x00])).unwrap(),
        );
        // Find SOF0.
        let markers = walk_markers(&jpeg);
        let mut i = 2; // after SOI
        let dqt_len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        i += 2 + dqt_len; // skip DQT
        assert_eq!(&jpeg[i..i + 2], &[MARKER_PREFIX, MARKER_SOF0]);
        let body = &jpeg[i + 4..];
        assert_eq!(body[0], 8, "8-bit precision");
        let height = u16::from_be_bytes([body[1], body[2]]);
        let width = u16::from_be_bytes([body[3], body[4]]);
        assert_eq!((width, height), (128, 96));
        assert_eq!(body[5], 3, "3 components");
        // Y component: id=1, sampling 0x22 (H=2 V=2), quant 0.
        assert_eq!(&body[6..9], &[1, 0x22, 0]);
        // Cb: id=2, 0x11, quant 1.
        assert_eq!(&body[9..12], &[2, 0x11, 1]);
        // Cr: id=3, 0x11, quant 1.
        assert_eq!(&body[12..15], &[3, 0x11, 1]);
        let _ = markers;
    }

    #[test]
    fn dht_carries_four_tables_with_correct_class_dest() {
        let jpeg = reconstruct_jpeg(
            &AmvVideoFrame::bind(&comedian_header(), &bare_frame(&[0x00])).unwrap(),
        );
        // Locate DHT by walking.
        let mut i = 2;
        for _ in 0..2 {
            // skip DQT then SOF0
            let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
            i += 2 + len;
        }
        assert_eq!(&jpeg[i..i + 2], &[MARKER_PREFIX, MARKER_DHT]);
        let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        let body = &jpeg[i + 4..i + 2 + len];
        // Walk the four sub-tables, collecting their Tc/Th bytes.
        let mut p = 0usize;
        let mut tc_th = Vec::new();
        for _ in 0..4 {
            tc_th.push(body[p]);
            // count HUFFVAL = sum of 16 BITS.
            let nvals: usize = body[p + 1..p + 17].iter().map(|&b| b as usize).sum();
            p += 1 + 16 + nvals;
        }
        assert_eq!(tc_th, vec![0x00, 0x10, 0x01, 0x11]);
        assert_eq!(p, body.len(), "DHT body fully consumed");
    }

    #[test]
    fn sos_single_interleaved_scan_full_spectral() {
        let jpeg = reconstruct_jpeg(
            &AmvVideoFrame::bind(&comedian_header(), &bare_frame(&[0x00])).unwrap(),
        );
        // SOS is the last header before entropy data.
        let mut i = 2;
        for _ in 0..3 {
            // DQT, SOF0, DHT
            let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
            i += 2 + len;
        }
        assert_eq!(&jpeg[i..i + 2], &[MARKER_PREFIX, MARKER_SOS]);
        let body = &jpeg[i + 4..];
        assert_eq!(body[0], 3, "3 components in scan");
        assert_eq!(&body[1..3], &[1, 0x00]); // Y Td=0 Ta=0
        assert_eq!(&body[3..5], &[2, 0x11]); // Cb Td=1 Ta=1
        assert_eq!(&body[5..7], &[3, 0x11]); // Cr Td=1 Ta=1
        assert_eq!(body[7], 0, "Ss=0");
        assert_eq!(body[8], 63, "Se=63");
        assert_eq!(body[9], 0, "Ah=Al=0");
    }

    #[test]
    fn entropy_and_eoi_preserved_verbatim() {
        let entropy = [0xE6, 0x49, 0xFF, 0x00, 0xA6, 0x93];
        let frame_bytes = bare_frame(&entropy);
        let jpeg = reconstruct_jpeg_from_payload(&comedian_header(), &frame_bytes).unwrap();
        // The output ends with the entropy bytes immediately followed
        // by EOI.
        let tail_start = jpeg.len() - entropy.len() - 2;
        assert_eq!(&jpeg[tail_start..jpeg.len() - 2], &entropy);
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]);
    }

    #[test]
    fn reconstruction_rejects_payload_with_internal_markers() {
        // bind_strict path must reject a payload carrying a DQT inside.
        let bad = [0xFF, 0xD8, 0xFF, 0xDB, 0x12, 0xFF, 0xD9];
        assert!(reconstruct_jpeg_from_payload(&comedian_header(), &bad).is_err());
    }

    #[test]
    fn header_segment_byte_count_is_constant() {
        // Every reconstructed frame prepends the same fixed header byte
        // count regardless of entropy length; compute it once.
        let a = reconstruct_jpeg(
            &AmvVideoFrame::bind(&comedian_header(), &bare_frame(&[0x00])).unwrap(),
        );
        let b = reconstruct_jpeg(
            &AmvVideoFrame::bind(&comedian_header(), &bare_frame(&[0x00, 0x11, 0x22])).unwrap(),
        );
        // Difference equals the difference in entropy length (2 bytes).
        assert_eq!(b.len() - a.len(), 2);
    }

    /// Real-fixture pin: reconstruct the first frame of `comedian.amv`
    /// and verify it is a structurally complete baseline JPEG with the
    /// canonical AMV header segments and the §4a first entropy bytes.
    #[test]
    fn comedian_first_frame_reconstructs_conforming_jpeg() {
        use crate::parse::{MoviPayload, MoviPayloadIter, AMV_END_TRAILER};

        let crate_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/comedian.amv");
        let workspace_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/container/amv/fixtures/comedian.amv");
        let path = if crate_path.exists() {
            crate_path
        } else if workspace_path.exists() {
            workspace_path
        } else {
            eprintln!("skipping comedian reconstruct fixture test: not staged");
            return;
        };
        let bytes = std::fs::read(&path).expect("read fixture");
        let header = AmvHeader::parse(&bytes[0x20..0x20 + AMVH_BODY_LEN as usize]).unwrap();

        let movi_pos = bytes.windows(4).position(|w| w == b"movi").unwrap();
        let trailer_start = bytes.len() - AMV_END_TRAILER.len();
        let movi_body = &bytes[movi_pos + 4..trailer_start];

        let first_video = MoviPayloadIter::new(movi_body)
            .filter_map(|r| r.ok())
            .find_map(|p| match p {
                MoviPayload::Video { body, .. } => Some(body.to_vec()),
                _ => None,
            })
            .expect("first video frame");
        assert_eq!(first_video.len(), 1633, "§4 first chunk size");

        let jpeg = reconstruct_jpeg_from_payload(&header, &first_video).expect("reconstructs");

        // Marker order intact.
        let markers = walk_markers(&jpeg);
        let tags: Vec<u8> = markers.iter().map(|&(m, _)| m).collect();
        assert_eq!(tags, vec![MARKER_DQT, MARKER_SOF0, MARKER_DHT, MARKER_SOS]);

        // Ends with the bare payload's EOI; entropy length preserved.
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]);
        // Reconstructed size = fixed header bytes + entropy(1629) + SOI(2) + EOI(2).
        // Entropy window of the §4a first frame is 1633 - 4 = 1629 bytes,
        // and begins E6 49 A6 93 right after the inserted SOS header.
        let sos_pos = jpeg
            .windows(2)
            .position(|w| w == [MARKER_PREFIX, MARKER_SOS])
            .unwrap();
        let sos_len = u16::from_be_bytes([jpeg[sos_pos + 2], jpeg[sos_pos + 3]]) as usize;
        let entropy_start = sos_pos + 2 + sos_len;
        assert_eq!(
            &jpeg[entropy_start..entropy_start + 4],
            &[0xE6, 0x49, 0xA6, 0x93]
        );
        assert_eq!(
            jpeg.len() - entropy_start - 2,
            1629,
            "entropy window length"
        );
    }
}
