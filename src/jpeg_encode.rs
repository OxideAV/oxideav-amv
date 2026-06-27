//! In-crate baseline-JPEG encode of RGB pixels to a bare AMV `00dc`
//! video frame — the forward (encoder) direction of the §4a device
//! profile in `docs/container/amv/amv-container-trace.md`.
//!
//! # Why this lives in the AMV crate
//!
//! This is the byte-inverse of [`crate::decode_frame`]. Per trace §4a
//! every AMV `00dc` payload is a **table-stripped** baseline JPEG: the
//! device's encoder DCT-codes the frame with the JPEG Annex K example
//! tables (K.1/K.2 quant, K.3/K.4 Huffman), 4:2:0 sampling, one
//! interleaved scan, then *removes* the `DQT` / `SOF0` / `DHT` / `SOS`
//! marker segments so the on-disk payload is `FF D8` + bare entropy +
//! `FF D9`. Reproducing that byte shape is intrinsic to the AMV device
//! profile (the wrapper has no other home — see `decode_frame`'s module
//! docs and `IMPLEMENTOR_ROUND.md` "Codecs with dedicated native
//! containers"), so the encoder belongs here next to the decoder.
//!
//! # The encode profile (the exact inverse of §4a decode)
//!
//! * Orientation — the encoder takes an **upright** RGB raster and emits
//!   the §4a **bottom-up** (DIB) coded order, the inverse of the decoder's
//!   post-decode vertical flip.
//! * Colour — BT.601 / JFIF RGB→YCbCr, the inverse of the decoder's
//!   `ycbcr_to_rgb`.
//! * Sampling — 4:2:0: each 16×16 MCU carries 4 luma blocks + 1 Cb + 1 Cr,
//!   chroma box-averaged 2×2 → one sample (the inverse of the decoder's
//!   nearest-neighbour 2× upsample).
//! * Transform — forward 8×8 DCT (the transpose-symmetric inverse of the
//!   decoder's `idct_8x8`), level-shift −128, quantize by K.1 (luma) /
//!   K.2 (chroma) with round-to-nearest.
//! * Entropy — canonical T.81 Huffman *encode* tables built from the same
//!   Annex K `BITS`/`HUFFVAL` lists the decoder walks; DC difference +
//!   predictor per component, AC run/size with ZRL + EOB, MSB-first bit
//!   writer with `FF`→`FF 00` byte stuffing (the inverse of the decoder's
//!   de-stuffing `BitReader`).
//!
//! The same Annex K table constants back both this encoder and the
//! decoder / reconstructor (shared from [`crate::jpeg_reconstruct`], not
//! duplicated). No JPEG/AMV **encoder** source was read — only the public
//! T.81 baseline algorithm and the public Annex K tables.

use crate::jpeg_decode::DecodedFrame;
use crate::jpeg_reconstruct::{
    AC_CHROMA_BITS, AC_CHROMA_VALS, AC_LUMA_BITS, AC_LUMA_VALS, DC_CHROMA_BITS, DC_CHROMA_VALS,
    DC_LUMA_BITS, DC_LUMA_VALS, QUANT_CHROMA, QUANT_LUMA, ZIGZAG,
};
use crate::video::flip_rows_vertical;
use crate::AmvDemuxerError;

// ---------------------------------------------------------------------
// Canonical Huffman ENCODE table (T.81 Annex C / K.2).
// ---------------------------------------------------------------------

/// A baseline Huffman *encode* table: for each `HUFFVAL` symbol, the
/// canonical `(code, length)` pair. Built from the same `BITS` / `HUFFVAL`
/// lists the decoder uses, by the canonical code-assignment of T.81
/// Annex C (shortest length first, increasing code).
struct HuffEncTable {
    /// `code[sym]` / `len[sym]`: the canonical Huffman code and its bit
    /// length for symbol value `sym` (0..=255). `len[sym] == 0` means the
    /// symbol is not in this table.
    code: [u16; 256],
    len: [u8; 256],
}

impl HuffEncTable {
    /// Build the encode table from `BITS` (16 length counts) + `HUFFVAL`.
    fn build(bits: &[u8; 16], huffval: &[u8]) -> Self {
        let mut code = [0u16; 256];
        let mut len = [0u8; 256];
        let mut next_code: u32 = 0;
        let mut k = 0usize;
        for (l_idx, &count) in bits.iter().enumerate() {
            let bit_len = (l_idx + 1) as u8;
            for _ in 0..count {
                let sym = huffval[k] as usize;
                code[sym] = next_code as u16;
                len[sym] = bit_len;
                next_code += 1;
                k += 1;
            }
            next_code <<= 1;
        }
        HuffEncTable { code, len }
    }
}

// ---------------------------------------------------------------------
// MSB-first bit writer with JPEG byte-stuffing (inverse of BitReader).
// ---------------------------------------------------------------------

/// MSB-first bit writer over the entropy-coded scan window. Re-applies
/// JPEG `FF`→`FF 00` byte stuffing so the emitted stream round-trips
/// through the decoder's de-stuffing [`crate::jpeg_decode`] `BitReader`.
struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    nbits: u32,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            out: Vec::new(),
            acc: 0,
            nbits: 0,
        }
    }

    /// Emit the low `len` bits of `code`, MSB-first.
    fn put(&mut self, code: u32, len: u32) {
        for i in (0..len).rev() {
            self.acc = (self.acc << 1) | ((code >> i) & 1);
            self.nbits += 1;
            if self.nbits == 8 {
                let b = (self.acc & 0xFF) as u8;
                self.out.push(b);
                if b == 0xFF {
                    self.out.push(0x00); // byte stuffing
                }
                self.nbits = 0;
                self.acc = 0;
            }
        }
    }

    /// Flush the final partial byte, padding the low bits with **1**s
    /// (T.81 §F.1.2.3: the trailing fill bits are 1s). Returns the stuffed
    /// entropy bytes.
    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.acc = (self.acc << pad) | ((1u32 << pad) - 1);
            let b = (self.acc & 0xFF) as u8;
            self.out.push(b);
            if b == 0xFF {
                self.out.push(0x00);
            }
        }
        self.out
    }
}

// ---------------------------------------------------------------------
// Forward DCT (separable 8×8, float — transpose of decode's idct_8x8).
// ---------------------------------------------------------------------

/// 8×8 separable forward DCT (T.81 §A.3.3), float reference. `block`
/// holds level-shifted spatial samples (already centered around 0) in
/// natural order on input; on output it holds the DCT coefficients
/// (natural order).
fn fdct_8x8(block: &mut [f32; 64]) {
    use std::f32::consts::PI;
    let mut tmp = [0f32; 64];
    // Columns: for each output frequency v, sum over spatial y.
    for x in 0..8usize {
        for v in 0..8usize {
            let cv = if v == 0 {
                std::f32::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let mut s = 0f32;
            for y in 0..8usize {
                s += block[y * 8 + x] * ((2 * y + 1) as f32 * v as f32 * PI / 16.0).cos();
            }
            tmp[v * 8 + x] = cv * s * 0.5;
        }
    }
    // Rows: for each output frequency u, sum over spatial x.
    for v in 0..8usize {
        for u in 0..8usize {
            let cu = if u == 0 {
                std::f32::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let mut s = 0f32;
            for x in 0..8usize {
                s += tmp[v * 8 + x] * ((2 * x + 1) as f32 * u as f32 * PI / 16.0).cos();
            }
            block[v * 8 + u] = cu * s * 0.5;
        }
    }
}

// ---------------------------------------------------------------------
// Colour + sampling (inverse of decode).
// ---------------------------------------------------------------------

/// BT.601 / JFIF full-range RGB → YCbCr — the inverse of the decoder's
/// `ycbcr_to_rgb`. Returns `(Y, Cb, Cr)` each on the 0..255 scale (Cb/Cr
/// centred at 128).
fn rgb_to_ycbcr(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
    let cr = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
    (y, cb, cr)
}

// ---------------------------------------------------------------------
// Block encode.
// ---------------------------------------------------------------------

/// Quantize the natural-order coefficients in `coeffs` by `quant` (round
/// to nearest) and Huffman-encode the block (DC difference + AC run/size)
/// into `w`. `pred` carries the running DC predictor for the component.
fn encode_block(
    w: &mut BitWriter,
    coeffs: &[f32; 64],
    quant: &[u8; 64],
    dc_tbl: &HuffEncTable,
    ac_tbl: &HuffEncTable,
    pred: &mut i32,
) {
    // Quantize into natural order, then read out in zig-zag order.
    let mut q = [0i32; 64];
    for (i, (&c, &qv)) in coeffs.iter().zip(quant.iter()).enumerate() {
        q[i] = (c / qv as f32).round() as i32;
    }

    // DC: difference against the predictor, encoded as category + bits.
    let dc = q[0];
    let diff = dc - *pred;
    *pred = dc;
    let (size, bits) = magnitude_category(diff);
    w.put(
        dc_tbl.code[size as usize] as u32,
        dc_tbl.len[size as usize] as u32,
    );
    if size > 0 {
        w.put(bits, size);
    }

    // AC: walk zig-zag positions 1..=63, run-length encoding zeros.
    let mut run = 0u32;
    for zz in 1..64usize {
        // Natural index whose zig-zag position is `zz`.
        let natural = ZIGZAG.iter().position(|&z| z as usize == zz).unwrap();
        let coeff = q[natural];
        if coeff == 0 {
            run += 1;
            continue;
        }
        // Emit ZRL (run/size 0xF0) for each full run of 16 zeros.
        while run >= 16 {
            w.put(ac_tbl.code[0xF0] as u32, ac_tbl.len[0xF0] as u32);
            run -= 16;
        }
        let (size, bits) = magnitude_category(coeff);
        let rs = ((run << 4) | size) as usize;
        w.put(ac_tbl.code[rs] as u32, ac_tbl.len[rs] as u32);
        w.put(bits, size);
        run = 0;
    }
    // Trailing zeros → EOB (run/size 0x00).
    if run > 0 {
        w.put(ac_tbl.code[0x00] as u32, ac_tbl.len[0x00] as u32);
    }
}

/// T.81 RECEIVE/EXTEND inverse: the magnitude category (`SSSS`) of a
/// signed coefficient and the `category`-bit value transmitted for it.
/// Category 0 (value 0) transmits no bits. A negative value of category
/// `s` transmits `value - 1` in `s` bits (the one's-complement form the
/// decoder's EXTEND reverses).
fn magnitude_category(value: i32) -> (u32, u32) {
    if value == 0 {
        return (0, 0);
    }
    let mag = value.unsigned_abs();
    let size = 32 - mag.leading_zeros();
    let bits = if value > 0 {
        value as u32
    } else {
        // Negative: low `size` bits of (value - 1).
        ((value - 1) as u32) & ((1u32 << size) - 1)
    };
    (size, bits)
}

// ---------------------------------------------------------------------
// Frame encode.
// ---------------------------------------------------------------------

/// Encode an upright RGB raster (`width`×`height`, packed `R,G,B`) into
/// the bare AMV `00dc` payload: `FF D8` + byte-stuffed entropy + `FF D9`,
/// table-stripped per §4a.
///
/// The same hardcoded device profile the decoder assumes (Annex K tables,
/// 4:2:0, single interleaved scan, bottom-up DIB order). Round-trips
/// through [`crate::decode_frame_from_payload`] as a stable fixed point:
/// the output decodes to a raster that re-encodes to the same bytes.
///
/// Returns `InvalidData` for a zero dimension or a `rgb` length that does
/// not equal `width * height * 3`.
pub fn encode_frame_rgb(width: u32, height: u32, rgb: &[u8]) -> Result<Vec<u8>, AmvDemuxerError> {
    if width == 0 || height == 0 {
        return Err(AmvDemuxerError::InvalidData(
            "AMV frame geometry must be non-zero".into(),
        ));
    }
    let w = width as usize;
    let h = height as usize;
    if rgb.len() != w * h * 3 {
        return Err(AmvDemuxerError::InvalidData(format!(
            "rgb length {} must equal width*height*3 = {}",
            rgb.len(),
            w * h * 3
        )));
    }

    // §4a inverse orientation: the device codes bottom-up (DIB) order, so
    // flip the upright raster to bottom-up before sampling. Work on a copy.
    let mut flipped = rgb.to_vec();
    flip_rows_vertical(&mut flipped, h, w * 3);

    // 16×16-MCU-aligned planes. Edge pixels are replicated into the pad
    // region so a partial final MCU codes without a hard edge (matches the
    // decoder, which crops the padded planes back to W×H).
    let mcus_x = w.div_ceil(16);
    let mcus_y = h.div_ceil(16);
    let luma_w = mcus_x * 16;
    let luma_h = mcus_y * 16;
    let chroma_w = mcus_x * 8;
    let chroma_h = mcus_y * 8;

    let mut y_plane = vec![0f32; luma_w * luma_h];
    let mut cb_plane = vec![0f32; chroma_w * chroma_h];
    let mut cr_plane = vec![0f32; chroma_w * chroma_h];

    // Fill luma + full-res chroma, replicating edges into the pad.
    let mut cb_full = vec![0f32; luma_w * luma_h];
    let mut cr_full = vec![0f32; luma_w * luma_h];
    for py in 0..luma_h {
        let sy = py.min(h - 1);
        for px in 0..luma_w {
            let sx = px.min(w - 1);
            let k = (sy * w + sx) * 3;
            let (yv, cb, cr) = rgb_to_ycbcr(
                flipped[k] as f32,
                flipped[k + 1] as f32,
                flipped[k + 2] as f32,
            );
            y_plane[py * luma_w + px] = yv;
            cb_full[py * luma_w + px] = cb;
            cr_full[py * luma_w + px] = cr;
        }
    }
    // 4:2:0 chroma: box-average each 2×2 luma-resolution region into one
    // chroma sample (inverse of the decoder's nearest 2× upsample).
    for cy in 0..chroma_h {
        for cx in 0..chroma_w {
            let mut sumb = 0f32;
            let mut sumr = 0f32;
            for dy in 0..2usize {
                for dx in 0..2usize {
                    let fy = cy * 2 + dy;
                    let fx = cx * 2 + dx;
                    sumb += cb_full[fy * luma_w + fx];
                    sumr += cr_full[fy * luma_w + fx];
                }
            }
            cb_plane[cy * chroma_w + cx] = sumb / 4.0;
            cr_plane[cy * chroma_w + cx] = sumr / 4.0;
        }
    }

    Ok(encode_planes_to_payload(
        &y_plane, luma_w, mcus_x, mcus_y, &cb_plane, &cr_plane, chroma_w,
    ))
}

/// Entropy-code the MCU grid of three already-prepared, MCU-pad-aligned
/// sample planes (luma at `luma_w`-stride, chroma at `chroma_w`-stride,
/// both holding §4a JFIF-range 0..255 values) into a bare `00dc`
/// payload. Shared by [`encode_frame_rgb`] and
/// [`encode_frame_yuv420p`]: only the plane-fill stage differs between
/// the RGB and native-YUV front doors; the DCT / quant / Huffman walk is
/// identical.
#[allow(clippy::too_many_arguments)]
fn encode_planes_to_payload(
    y_plane: &[f32],
    luma_w: usize,
    mcus_x: usize,
    mcus_y: usize,
    cb_plane: &[f32],
    cr_plane: &[f32],
    chroma_w: usize,
) -> Vec<u8> {
    // Build the four Annex K Huffman encode tables once.
    let dc_luma = HuffEncTable::build(&DC_LUMA_BITS, &DC_LUMA_VALS);
    let ac_luma = HuffEncTable::build(&AC_LUMA_BITS, &AC_LUMA_VALS);
    let dc_chroma = HuffEncTable::build(&DC_CHROMA_BITS, &DC_CHROMA_VALS);
    let ac_chroma = HuffEncTable::build(&AC_CHROMA_BITS, &AC_CHROMA_VALS);

    let mut bw = BitWriter::new();
    let mut pred_y = 0i32;
    let mut pred_cb = 0i32;
    let mut pred_cr = 0i32;

    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            // Four luma blocks (raster order within the MCU).
            for by in 0..2usize {
                for bx in 0..2usize {
                    let ox = mx * 16 + bx * 8;
                    let oy = my * 16 + by * 8;
                    let mut blk = gather_block(y_plane, luma_w, ox, oy);
                    fdct_8x8(&mut blk);
                    encode_block(&mut bw, &blk, &QUANT_LUMA, &dc_luma, &ac_luma, &mut pred_y);
                }
            }
            // One Cb, one Cr block.
            let cox = mx * 8;
            let coy = my * 8;
            let mut cb_blk = gather_block(cb_plane, chroma_w, cox, coy);
            fdct_8x8(&mut cb_blk);
            encode_block(
                &mut bw,
                &cb_blk,
                &QUANT_CHROMA,
                &dc_chroma,
                &ac_chroma,
                &mut pred_cb,
            );
            let mut cr_blk = gather_block(cr_plane, chroma_w, cox, coy);
            fdct_8x8(&mut cr_blk);
            encode_block(
                &mut bw,
                &cr_blk,
                &QUANT_CHROMA,
                &dc_chroma,
                &ac_chroma,
                &mut pred_cr,
            );
        }
    }

    let entropy = bw.finish();
    let mut payload = Vec::with_capacity(entropy.len() + 4);
    payload.extend_from_slice(&[0xFF, 0xD8]); // SOI
    payload.extend_from_slice(&entropy);
    payload.extend_from_slice(&[0xFF, 0xD9]); // EOI
    payload
}

/// Encode native planar **YUV420P** (the [`crate::DecodedYuv420p`] shape)
/// straight into a bare AMV `00dc` payload — the exact byte-inverse of
/// [`crate::decode_frame_yuv420p`], with no YCbCr↔RGB round-trip.
///
/// `y` is `width * height` bytes; `cb` / `cr` are each
/// `ceil(width/2) * ceil(height/2)` bytes (4:2:0). All three are
/// **upright** (the §4a bottom-up flip is applied here on the way in,
/// per plane). Edge samples replicate into the 16×16-MCU pad so a
/// partial final MCU codes cleanly, matching the decoder's crop. Both
/// chroma planes are nearest-upsampled to luma resolution then
/// box-averaged back, so encode∘decode is the same fixed point the RGB
/// path reaches.
///
/// Returns `InvalidData` for a zero dimension or a plane length that
/// does not match the 4:2:0 geometry.
pub fn encode_frame_yuv420p(
    width: u32,
    height: u32,
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
) -> Result<Vec<u8>, AmvDemuxerError> {
    if width == 0 || height == 0 {
        return Err(AmvDemuxerError::InvalidData(
            "AMV frame geometry must be non-zero".into(),
        ));
    }
    let w = width as usize;
    let h = height as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    if y.len() != w * h {
        return Err(AmvDemuxerError::InvalidData(format!(
            "y length {} must equal width*height = {}",
            y.len(),
            w * h
        )));
    }
    if cb.len() != cw * ch || cr.len() != cw * ch {
        return Err(AmvDemuxerError::InvalidData(format!(
            "chroma length must equal ceil(w/2)*ceil(h/2) = {} (cb={}, cr={})",
            cw * ch,
            cb.len(),
            cr.len()
        )));
    }

    let mcus_x = w.div_ceil(16);
    let mcus_y = h.div_ceil(16);
    let luma_w = mcus_x * 16;
    let luma_h = mcus_y * 16;
    let chroma_w = mcus_x * 8;
    let chroma_h = mcus_y * 8;

    let mut y_plane = vec![0f32; luma_w * luma_h];
    let mut cb_plane = vec![0f32; chroma_w * chroma_h];
    let mut cr_plane = vec![0f32; chroma_w * chroma_h];

    // §4a inverse orientation: code bottom-up. The source planes are
    // upright, so plane row `py` (top-down, bottom-up coded) samples
    // upright source row `h - 1 - py` (luma) / `ch - 1 - cy` (chroma),
    // clamped + edge-replicated into the MCU pad.
    for py in 0..luma_h {
        let up = (h - 1).saturating_sub(py.min(h - 1));
        for px in 0..luma_w {
            let sx = px.min(w - 1);
            y_plane[py * luma_w + px] = y[up * w + sx] as f32;
        }
    }
    for cy in 0..chroma_h {
        let up = (ch - 1).saturating_sub(cy.min(ch - 1));
        for cx in 0..chroma_w {
            let sx = cx.min(cw - 1);
            cb_plane[cy * chroma_w + cx] = cb[up * cw + sx] as f32;
            cr_plane[cy * chroma_w + cx] = cr[up * cw + sx] as f32;
        }
    }

    Ok(encode_planes_to_payload(
        &y_plane, luma_w, mcus_x, mcus_y, &cb_plane, &cr_plane, chroma_w,
    ))
}

/// Gather an 8×8 block from `plane` (width `plane_w`) at top-left
/// `(ox, oy)`, level-shifting by −128 into a float block (natural order).
fn gather_block(plane: &[f32], plane_w: usize, ox: usize, oy: usize) -> [f32; 64] {
    let mut blk = [0f32; 64];
    for ty in 0..8usize {
        for tx in 0..8usize {
            blk[ty * 8 + tx] = plane[(oy + ty) * plane_w + (ox + tx)] - 128.0;
        }
    }
    blk
}

/// Encode a [`DecodedFrame`] (the decoder's output type) back into a bare
/// AMV `00dc` payload — the convenience that pairs decode→encode.
pub fn encode_frame(frame: &DecodedFrame) -> Result<Vec<u8>, AmvDemuxerError> {
    encode_frame_rgb(frame.width, frame.height, &frame.rgb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg_decode::decode_frame_from_payload;
    use crate::parse::AmvHeader;

    fn header_wh(width: u32, height: u32) -> AmvHeader {
        AmvHeader {
            micros_per_frame: 83_333,
            width,
            height,
            fps: 12,
            flag_one: 1,
            reserved_30: 0,
            duration_packed: 0,
        }
    }

    #[test]
    fn huff_encode_table_matches_canonical_codes() {
        // K.3 luma DC: BITS [0,1,5,...]. Symbol HUFFVAL[0]=0 is the single
        // length-2 code → 0b00; HUFFVAL[1..]=1..5 are length-3 codes
        // 010,011,100,101,110.
        let t = HuffEncTable::build(&DC_LUMA_BITS, &DC_LUMA_VALS);
        assert_eq!((t.code[0], t.len[0]), (0b00, 2));
        assert_eq!((t.code[1], t.len[1]), (0b010, 3));
        assert_eq!((t.code[3], t.len[3]), (0b100, 3));
        assert_eq!((t.code[5], t.len[5]), (0b110, 3));
        // A symbol not in the table has len 0.
        assert_eq!(t.len[200], 0);
    }

    #[test]
    fn magnitude_category_round_trips_through_extend() {
        // Mirror the decoder's EXTEND: a positive v of category s is its
        // low s bits; a negative v is (v-1) in s bits.
        // size=3, +4 → (3, 0b100).
        assert_eq!(magnitude_category(4), (3, 0b100));
        // size=3, -4 → (3, 0b011) (the decoder reverses: 3 - 8 + 1 = -4).
        assert_eq!(magnitude_category(-4), (3, 0b011));
        // 0 → category 0, no bits.
        assert_eq!(magnitude_category(0), (0, 0));
        // +1 → (1, 1); -1 → (1, 0).
        assert_eq!(magnitude_category(1), (1, 1));
        assert_eq!(magnitude_category(-1), (1, 0));
    }

    #[test]
    fn fdct_then_idct_is_near_identity() {
        // The forward DCT must be the inverse of the decoder's IDCT to
        // round-trip. Encode a known spatial block, IDCT it back.
        // (Re-implement the decoder's IDCT inline to avoid exposing it.)
        let mut spatial = [0f32; 64];
        for (i, s) in spatial.iter_mut().enumerate() {
            *s = ((i * 7 % 100) as f32) - 50.0;
        }
        let mut coeffs = spatial;
        fdct_8x8(&mut coeffs);
        // Inverse via the same separable basis.
        use std::f32::consts::PI;
        let mut tmp = [0f32; 64];
        for (y, row) in tmp.chunks_exact_mut(8).enumerate() {
            for (x, out) in row.iter_mut().enumerate() {
                let mut s = 0f32;
                for u in 0..8usize {
                    let cu = if u == 0 {
                        std::f32::consts::FRAC_1_SQRT_2
                    } else {
                        1.0
                    };
                    s += cu * coeffs[y * 8 + u] * ((2 * x + 1) as f32 * u as f32 * PI / 16.0).cos();
                }
                *out = s * 0.5;
            }
        }
        let mut back = [0f32; 64];
        for x in 0..8usize {
            for y in 0..8usize {
                let mut s = 0f32;
                for v in 0..8usize {
                    let cv = if v == 0 {
                        std::f32::consts::FRAC_1_SQRT_2
                    } else {
                        1.0
                    };
                    s += cv * tmp[v * 8 + x] * ((2 * y + 1) as f32 * v as f32 * PI / 16.0).cos();
                }
                back[y * 8 + x] = s * 0.5;
            }
        }
        for (a, b) in spatial.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-2, "fDCT∘IDCT not identity: {a} vs {b}");
        }
    }

    #[test]
    fn rejects_bad_geometry_and_length() {
        assert!(encode_frame_rgb(0, 8, &[]).is_err());
        assert!(encode_frame_rgb(8, 8, &[0u8; 10]).is_err());
    }

    #[test]
    fn flat_frame_encodes_and_decodes_to_same_color() {
        // A uniform mid-gray frame must encode to a payload that decodes
        // back to (very near) the same uniform color.
        for (w, h) in [(16u32, 16u32), (128, 96)] {
            let rgb = vec![128u8; (w * h * 3) as usize];
            let payload = encode_frame_rgb(w, h, &rgb).expect("encode");
            assert_eq!(&payload[..2], &[0xFF, 0xD8]);
            assert_eq!(&payload[payload.len() - 2..], &[0xFF, 0xD9]);
            let frame = decode_frame_from_payload(&header_wh(w, h), &payload).expect("decode");
            assert_eq!((frame.width, frame.height), (w, h));
            // Uniform input → uniform output at the same level.
            assert!(
                frame.rgb.iter().all(|&b| b.abs_diff(128) <= 1),
                "{w}×{h}: flat frame must round-trip to ~128"
            );
        }
    }

    #[test]
    fn encode_decode_is_a_stable_fixed_point() {
        // A synthetic gradient: encode → decode → re-encode must reproduce
        // identical payload bytes (the canonical JPEG fixed point), and the
        // second decode equals the first.
        let (w, h) = (32u32, 32u32);
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let k = ((y * w + x) * 3) as usize;
                rgb[k] = (x * 8) as u8;
                rgb[k + 1] = (y * 8) as u8;
                rgb[k + 2] = ((x + y) * 4) as u8;
            }
        }
        let payload1 = encode_frame_rgb(w, h, &rgb).expect("encode 1");
        let decoded1 = decode_frame_from_payload(&header_wh(w, h), &payload1).expect("decode 1");
        let payload2 = encode_frame(&decoded1).expect("encode 2");
        let decoded2 = decode_frame_from_payload(&header_wh(w, h), &payload2).expect("decode 2");
        assert_eq!(
            payload1, payload2,
            "re-encoding the decoded raster reproduces identical bytes"
        );
        assert_eq!(decoded1.rgb, decoded2.rgb, "decode is a stable fixed point");
    }

    #[test]
    fn round_trip_preserves_image_structure() {
        // A structured natural-ish image (smooth gradients) must survive
        // the lossy round-trip with low mean abs error per channel.
        let (w, h) = (64u32, 48u32);
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let k = ((y * w + x) * 3) as usize;
                let fx = x as f64 / w as f64;
                let fy = y as f64 / h as f64;
                rgb[k] = (200.0 * fx + 30.0) as u8;
                rgb[k + 1] = (180.0 * fy + 40.0) as u8;
                rgb[k + 2] = (120.0 * (fx + fy) / 2.0 + 60.0) as u8;
            }
        }
        let payload = encode_frame_rgb(w, h, &rgb).expect("encode");
        let frame = decode_frame_from_payload(&header_wh(w, h), &payload).expect("decode");
        let mae: f64 = rgb
            .iter()
            .zip(&frame.rgb)
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / rgb.len() as f64;
        // Annex K "quality ~50" tables over a smooth image: a few levels.
        assert!(mae < 8.0, "round-trip MAE {mae}/channel too high");
    }

    #[test]
    fn payload_passes_decoder_strict_bind() {
        // The encoder output must be a valid bare §4a frame (SOI/EOI
        // bracket, no internal markers) — i.e. strict-bind must accept it.
        let (w, h) = (48u32, 32u32);
        let rgb = vec![100u8; (w * h * 3) as usize];
        let payload = encode_frame_rgb(w, h, &rgb).expect("encode");
        // bind_strict runs the §4a no-internal-markers scan.
        let frame = crate::video::AmvVideoFrame::bind_strict(&header_wh(w, h), &payload)
            .expect("encoder output is a valid bare §4a frame");
        assert_eq!((frame.width(), frame.height()), (w, h));
    }

    #[test]
    fn non_mod16_geometry_round_trips() {
        // Non-multiple-of-16 dimensions must encode (edge replication into
        // the MCU pad) and decode back to the right geometry without
        // padding leaking into the crop.
        for (w, h) in [(17u32, 17u32), (20, 12), (33, 9), (1, 1)] {
            let rgb = vec![140u8; (w * h * 3) as usize];
            let payload = encode_frame_rgb(w, h, &rgb).expect("encode non-mod16");
            let frame = decode_frame_from_payload(&header_wh(w, h), &payload).expect("decode");
            assert_eq!((frame.width, frame.height), (w, h));
            assert!(
                frame.rgb.iter().all(|&b| b.abs_diff(140) <= 2),
                "{w}×{h}: flat non-mod16 frame round-trips"
            );
        }
    }

    #[test]
    fn yuv420p_encode_matches_rgb_encode_byte_for_byte() {
        // Decode the RGB-path output of a flat frame to native YUV planes,
        // then re-encode via the YUV front door: it must produce the
        // identical bare payload the RGB path does on the same content
        // (both reach the §4a fixed point). This proves the native-YUV
        // encode and the RGB encode share one quantized representation.
        use crate::jpeg_decode::decode_frame_yuv420p_from_payload;
        for (w, h) in [(16u32, 16u32), (17, 17), (33, 9), (32, 32)] {
            let rgb = vec![123u8; (w * h * 3) as usize];
            let rgb_payload = encode_frame_rgb(w, h, &rgb).expect("rgb encode");
            let yuv = decode_frame_yuv420p_from_payload(&header_wh(w, h), &rgb_payload)
                .expect("decode to yuv");
            let yuv_payload =
                encode_frame_yuv420p(w, h, &yuv.y, &yuv.cb, &yuv.cr).expect("yuv encode");
            // Re-encoding the decoded planes is a fixed point with the RGB
            // path: decode(rgb_payload) → planes → encode == rgb_payload.
            assert_eq!(
                yuv_payload, rgb_payload,
                "{w}×{h}: native-YUV re-encode must equal the RGB-path payload"
            );
        }
    }

    #[test]
    fn yuv420p_encode_decode_is_a_stable_fixed_point() {
        // encode_yuv → decode_yuv → encode_yuv must be byte-stable on a
        // structured (non-flat) frame: a per-MCU luma gradient with
        // mid-level chroma.
        let (w, h) = (33u32, 17u32);
        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let mut y = vec![0u8; (w * h) as usize];
        for yy in 0..h as usize {
            for xx in 0..w as usize {
                y[yy * w as usize + xx] = (((xx / 16 + yy / 16) * 40 + 40) % 256) as u8;
            }
        }
        let cb = vec![128u8; cw * ch];
        let cr = vec![140u8; cw * ch];

        let p1 = encode_frame_yuv420p(w, h, &y, &cb, &cr).expect("encode 1");
        let dec = crate::jpeg_decode::decode_frame_yuv420p_from_payload(&header_wh(w, h), &p1)
            .expect("decode");
        let p2 = encode_frame_yuv420p(w, h, &dec.y, &dec.cb, &dec.cr).expect("re-encode");
        assert_eq!(p1, p2, "encode∘decode∘encode must be a stable fixed point");
    }

    #[test]
    fn yuv420p_encode_rejects_bad_lengths() {
        assert!(encode_frame_yuv420p(0, 16, &[], &[], &[]).is_err());
        // y too short
        assert!(encode_frame_yuv420p(16, 16, &[0u8; 100], &[0u8; 64], &[0u8; 64]).is_err());
        // chroma wrong
        assert!(encode_frame_yuv420p(16, 16, &[0u8; 256], &[0u8; 10], &[0u8; 64]).is_err());
    }
}
