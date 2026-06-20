//! In-crate baseline-JPEG decode of an AMV `00dc` video frame to RGB
//! pixels — the §4a "decoded raster is a coherent natural image" step of
//! `docs/container/amv/amv-container-trace.md`.
//!
//! # Why this lives in the AMV crate
//!
//! AMV's video is **not** a generic JPEG carried in a generic container:
//! per trace §4a every `00dc` payload is a table-stripped bitstream whose
//! quantization tables, Huffman tables, frame geometry and scan
//! parameters are **all absent from the wire** and hardcoded in the
//! player's decoder. The resolution comes only from the §2 `amvh`
//! header. There is no `fccHandler`, no `BITMAPINFOHEADER`, no `DQT` /
//! `SOF0` / `DHT` / `SOS` — "Every AMV file has the same wrapper", so the
//! fixed device profile is intrinsic to the format. Decoding it therefore
//! belongs in this crate (crate-purpose discipline: the AMV wrapper has no
//! other reasonable home — see `docs/IMPLEMENTOR_ROUND.md` "Codecs with
//! dedicated native containers").
//!
//! [`reconstruct_jpeg`](crate::reconstruct_jpeg) splices the stripped
//! marker segments back to produce a conforming JPEG for a downstream
//! generic decoder. This module is the *self-contained* path: it decodes
//! the bare entropy stream straight to pixels using the §4a device tables,
//! with no external binary and no synthesised intermediate JPEG.
//!
//! # The §4a device profile (all reconstruction-verified in the trace)
//!
//! * Quantization — Annex K K.1 (luma) / K.2 (chroma), 8-bit, unscaled.
//! * Frame — baseline sequential DCT, 8-bit, 3 components Y/Cb/Cr,
//!   **4:2:0** (luma `2×2`, chroma `1×1`), W×H from `amvh`.
//! * Huffman — Annex K K.3 (luma DC+AC) / K.4 (chroma DC+AC), verbatim.
//! * Scan — single interleaved scan, `Ss=0 Se=63 Ah=0 Al=0`.
//! * Orientation — bottom-up (DIB row order); the decoded raster is
//!   vertically mirrored and a single flip yields the upright image.
//!
//! The implementation is a from-scratch baseline decoder: a bit reader
//! over the byte-stuffed entropy window, a canonical-Huffman walk built
//! from the Annex K `BITS`/`HUFFVAL` lists, zig-zag dequantization, an
//! 8×8 separable inverse DCT, 4:2:0 chroma upsampling and a BT.601
//! YCbCr→RGB conversion. No JPEG/AMV **decoder** source was read — only
//! the public T.81 baseline algorithm and the public Annex K tables (the
//! same table data already used by [`crate::reconstruct_jpeg`]).

use crate::jpeg_reconstruct::{
    AC_CHROMA_BITS, AC_CHROMA_VALS, AC_LUMA_BITS, AC_LUMA_VALS, DC_CHROMA_BITS, DC_CHROMA_VALS,
    DC_LUMA_BITS, DC_LUMA_VALS, QUANT_CHROMA, QUANT_LUMA, ZIGZAG,
};
use crate::parse::AmvHeader;
use crate::video::{flip_rows_vertical, AmvVideoFrame};
use crate::AmvDemuxerError;

/// A decoded AMV video frame: an upright, 8-bit interleaved RGB raster at
/// the §2 `amvh` resolution.
///
/// Row order is top-to-bottom (the §4a bottom-up DIB mirroring has
/// already been corrected — see [`decode_frame`]); pixels are packed
/// `R, G, B` left-to-right. `rgb.len() == width * height * 3`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedFrame {
    /// Frame width in pixels (§2 `amvh` +0x20).
    pub width: u32,
    /// Frame height in pixels (§2 `amvh` +0x24).
    pub height: u32,
    /// Interleaved 8-bit `R, G, B` bytes, row-major top-to-bottom.
    pub rgb: Vec<u8>,
}

impl DecodedFrame {
    /// The BT.601 luma value of pixel `(x, y)` (0-based, `y` from the
    /// top). Convenience for tests / coherence checks; returns `0.0` for
    /// out-of-range coordinates.
    pub fn luma_at(&self, x: usize, y: usize) -> f64 {
        if x >= self.width as usize || y >= self.height as usize {
            return 0.0;
        }
        let k = (y * self.width as usize + x) * 3;
        0.299 * self.rgb[k] as f64 + 0.587 * self.rgb[k + 1] as f64 + 0.114 * self.rgb[k + 2] as f64
    }
}

// ---------------------------------------------------------------------
// Canonical Huffman decode table (T.81 Annex C / F.2.2.3).
// ---------------------------------------------------------------------

/// A baseline Huffman table built from a `BITS` (16 length counts) +
/// `HUFFVAL` list. Decoding is the canonical T.81 procedure: assign codes
/// in increasing length, shortest first.
struct HuffTable {
    /// `min_code[l]` / `max_code[l]` bound the numeric range of codes of
    /// length `l` (1..=16); `val_ptr[l]` indexes the first `HUFFVAL`
    /// entry for that length. `max_code[l] < 0` marks "no codes of this
    /// length".
    min_code: [i32; 17],
    max_code: [i32; 17],
    val_ptr: [usize; 17],
    huffval: Vec<u8>,
}

impl HuffTable {
    /// Build the decode table from `BITS` (length counts for lengths
    /// 1..=16, stored 0-based so `bits[l-1]` is the count for length `l`)
    /// and `huffval`.
    fn build(bits: &[u8; 16], huffval: &[u8]) -> Self {
        // Generate HUFFSIZE / HUFFCODE (T.81 Annex C.2).
        let mut huffsize = Vec::with_capacity(huffval.len() + 1);
        for (l, &count) in bits.iter().enumerate() {
            for _ in 0..count {
                huffsize.push((l + 1) as u8);
            }
        }
        let mut min_code = [0i32; 17];
        let mut max_code = [-1i32; 17];
        let mut val_ptr = [0usize; 17];
        let mut code: i32 = 0;
        let mut k = 0usize; // index into huffsize / huffval
        let mut length = if huffsize.is_empty() { 0 } else { huffsize[0] };
        while k < huffsize.len() {
            // All codes of the current `length`.
            val_ptr[length as usize] = k;
            min_code[length as usize] = code;
            while k < huffsize.len() && huffsize[k] == length {
                code += 1;
                k += 1;
            }
            max_code[length as usize] = code - 1;
            code <<= 1;
            if k < huffsize.len() {
                // Advance to the next present length, shifting `code`
                // once per skipped (empty) length.
                let next = huffsize[k];
                while length < next {
                    length += 1;
                    if length != next {
                        code <<= 1;
                    }
                }
            }
        }
        HuffTable {
            min_code,
            max_code,
            val_ptr,
            huffval: huffval.to_vec(),
        }
    }
}

// ---------------------------------------------------------------------
// Entropy bit reader over the byte-stuffed scan window (T.81 §F.2.2.5).
// ---------------------------------------------------------------------

/// MSB-first bit reader over the entropy-coded scan window. Handles JPEG
/// byte-stuffing (`FF 00` → literal `FF`) and treats any other marker
/// (`FF xx`, xx != 0) — or running off the end — as an end-of-data
/// condition that yields zero bits (matching a baseline decoder padding a
/// truncated final MCU with zero coefficients).
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buf: u32,
    bit_cnt: u32,
    /// Set once a marker / EOF has been hit; further reads return 0.
    exhausted: bool,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            pos: 0,
            bit_buf: 0,
            bit_cnt: 0,
            exhausted: false,
        }
    }

    /// Pull one byte of entropy data, resolving byte-stuffing. Returns
    /// `None` at a real marker or EOF. Once a marker / EOF is reached the
    /// reader latches `exhausted` so subsequent calls stay at end-of-data
    /// (they must not skip the marker and resume reading the trailing
    /// marker payload as entropy bytes).
    fn next_byte(&mut self) -> Option<u8> {
        if self.exhausted {
            return None;
        }
        if self.pos >= self.data.len() {
            self.exhausted = true;
            return None;
        }
        let b = self.data[self.pos];
        if b != 0xFF {
            self.pos += 1;
            return Some(b);
        }
        // 0xFF: look at the following byte (without consuming the FF yet).
        if self.pos + 1 >= self.data.len() {
            self.exhausted = true;
            return None;
        }
        let n = self.data[self.pos + 1];
        if n == 0x00 {
            // Stuffed FF → literal 0xFF; consume both bytes.
            self.pos += 2;
            Some(0xFF)
        } else {
            // A real marker — entropy data ends here; latch end-of-data.
            self.exhausted = true;
            None
        }
    }

    /// Read a single bit (MSB-first). Past end-of-data, returns 0 (the
    /// final partial MCU is zero-padded).
    fn read_bit(&mut self) -> u32 {
        if self.bit_cnt == 0 {
            match self.next_byte() {
                Some(b) => {
                    self.bit_buf = b as u32;
                    self.bit_cnt = 8;
                }
                None => {
                    self.exhausted = true;
                    return 0;
                }
            }
        }
        self.bit_cnt -= 1;
        (self.bit_buf >> self.bit_cnt) & 1
    }

    /// Read `n` bits as an unsigned MSB-first integer.
    fn read_bits(&mut self, n: u32) -> u32 {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.read_bit();
        }
        v
    }

    /// Decode one Huffman-coded symbol via the canonical length walk
    /// (T.81 §F.2.2.3): accumulate bits until `code <= max_code[len]`.
    fn decode_huff(&mut self, table: &HuffTable) -> u8 {
        let mut code: i32 = 0;
        for len in 1..=16usize {
            code = (code << 1) | self.read_bit() as i32;
            if self.exhausted {
                return 0;
            }
            if table.max_code[len] >= 0 && code <= table.max_code[len] {
                let idx = table.val_ptr[len] + (code - table.min_code[len]) as usize;
                if idx < table.huffval.len() {
                    return table.huffval[idx];
                }
                return 0;
            }
        }
        0
    }

    /// Extend a `size`-bit magnitude to a signed coefficient
    /// (T.81 §F.2.2.1 / "RECEIVE" + "EXTEND").
    fn receive_extend(&mut self, size: u32) -> i32 {
        if size == 0 {
            return 0;
        }
        let v = self.read_bits(size) as i32;
        // EXTEND: if the high bit is 0 the value is negative.
        if v < (1 << (size - 1)) {
            v - (1 << size) + 1
        } else {
            v
        }
    }
}

// ---------------------------------------------------------------------
// Inverse DCT (separable 8×8, floating point — AAN-free reference form).
// ---------------------------------------------------------------------

/// 8×8 separable inverse DCT (T.81 §A.3.3), float reference. `block`
/// holds dequantized coefficients in natural (row-major) order on input;
/// on output it holds the spatial-domain samples (still centered around
/// 0; the +128 level shift is applied by the caller).
fn idct_8x8(block: &mut [f32; 64]) {
    // Precomputed cosine basis: c[u][x] = cos((2x+1)·u·π/16) · alpha(u),
    // where alpha(0) = 1/√2, alpha(u>0) = 1.
    use std::f32::consts::PI;
    // Rows, then columns (separable).
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
                s += cu * block[y * 8 + u] * ((2 * x + 1) as f32 * u as f32 * PI / 16.0).cos();
            }
            *out = s * 0.5;
        }
    }
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
            block[y * 8 + x] = s * 0.5;
        }
    }
}

// ---------------------------------------------------------------------
// Component scaffolding.
// ---------------------------------------------------------------------

/// A decoded component plane at its own (subsampled) resolution. Height
/// is implied by `samples.len() / width`; only `width` is needed for the
/// row-stride into `samples`.
struct Plane {
    width: usize,
    samples: Vec<u8>,
}

/// Decode one 8×8 block from the entropy stream into a dequantized,
/// IDCT'd, level-shifted 8×8 sample tile (natural order). `pred` carries
/// the running DC predictor for this component (updated in place).
#[allow(clippy::too_many_arguments)]
fn decode_block(
    br: &mut BitReader<'_>,
    dc_tbl: &HuffTable,
    ac_tbl: &HuffTable,
    quant: &[u8; 64],
    pred: &mut i32,
) -> [u8; 64] {
    let mut coeffs = [0f32; 64];

    // DC: difference + predictor, dequantized into natural position 0.
    let t = br.decode_huff(dc_tbl) as u32;
    let diff = br.receive_extend(t);
    *pred += diff;
    coeffs[0] = (*pred * quant[0] as i32) as f32;

    // AC: run-length of zeros + magnitude, walking zig-zag positions.
    let mut k = 1usize;
    while k < 64 {
        let rs = br.decode_huff(ac_tbl);
        let run = (rs >> 4) as usize;
        let size = (rs & 0x0F) as u32;
        if size == 0 {
            if run == 15 {
                // ZRL: skip 16 zeros.
                k += 16;
                continue;
            }
            break; // EOB
        }
        k += run;
        if k >= 64 {
            break;
        }
        let level = br.receive_extend(size);
        let natural = ZIGZAG.iter().position(|&z| z as usize == k).unwrap();
        coeffs[natural] = (level * quant[natural] as i32) as f32;
        k += 1;
    }

    idct_8x8(&mut coeffs);

    let mut out = [0u8; 64];
    for (o, c) in out.iter_mut().zip(coeffs.iter()) {
        let v = (c.round() as i32) + 128;
        *o = v.clamp(0, 255) as u8;
    }
    out
}

/// Decode the entropy window of a §4a-bound AMV frame into Y/Cb/Cr
/// planes. `width`/`height` are the §2 `amvh` geometry.
fn decode_planes(
    entropy: &[u8],
    width: usize,
    height: usize,
) -> Result<(Plane, Plane, Plane), AmvDemuxerError> {
    if width == 0 || height == 0 {
        return Err(AmvDemuxerError::InvalidData(
            "AMV frame geometry must be non-zero".into(),
        ));
    }

    // Build the four Annex K Huffman decode tables once.
    let dc_luma = HuffTable::build(&DC_LUMA_BITS, &DC_LUMA_VALS);
    let ac_luma = HuffTable::build(&AC_LUMA_BITS, &AC_LUMA_VALS);
    let dc_chroma = HuffTable::build(&DC_CHROMA_BITS, &DC_CHROMA_VALS);
    let ac_chroma = HuffTable::build(&AC_CHROMA_BITS, &AC_CHROMA_VALS);

    // 4:2:0: each MCU is 16×16 luma = 2×2 luma blocks + 1 Cb + 1 Cr.
    let mcus_x = width.div_ceil(16);
    let mcus_y = height.div_ceil(16);
    let luma_w = mcus_x * 16;
    let luma_h = mcus_y * 16;
    let chroma_w = mcus_x * 8;
    let chroma_h = mcus_y * 8;

    let mut y_plane = vec![0u8; luma_w * luma_h];
    let mut cb_plane = vec![0u8; chroma_w * chroma_h];
    let mut cr_plane = vec![0u8; chroma_w * chroma_h];

    let mut br = BitReader::new(entropy);
    let mut pred_y = 0i32;
    let mut pred_cb = 0i32;
    let mut pred_cr = 0i32;

    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            // Four luma blocks in raster order within the MCU.
            for by in 0..2usize {
                for bx in 0..2usize {
                    let tile = decode_block(&mut br, &dc_luma, &ac_luma, &QUANT_LUMA, &mut pred_y);
                    let ox = mx * 16 + bx * 8;
                    let oy = my * 16 + by * 8;
                    blit_tile(&mut y_plane, luma_w, ox, oy, &tile);
                }
            }
            // One Cb, one Cr block.
            let cb_tile =
                decode_block(&mut br, &dc_chroma, &ac_chroma, &QUANT_CHROMA, &mut pred_cb);
            let cr_tile =
                decode_block(&mut br, &dc_chroma, &ac_chroma, &QUANT_CHROMA, &mut pred_cr);
            let cox = mx * 8;
            let coy = my * 8;
            blit_tile(&mut cb_plane, chroma_w, cox, coy, &cb_tile);
            blit_tile(&mut cr_plane, chroma_w, cox, coy, &cr_tile);
        }
    }

    Ok((
        Plane {
            width: luma_w,
            samples: y_plane,
        },
        Plane {
            width: chroma_w,
            samples: cb_plane,
        },
        Plane {
            width: chroma_w,
            samples: cr_plane,
        },
    ))
}

/// Copy an 8×8 tile into `plane` (width `plane_w`) at top-left `(ox, oy)`.
fn blit_tile(plane: &mut [u8], plane_w: usize, ox: usize, oy: usize, tile: &[u8; 64]) {
    for ty in 0..8usize {
        let dst = (oy + ty) * plane_w + ox;
        plane[dst..dst + 8].copy_from_slice(&tile[ty * 8..ty * 8 + 8]);
    }
}

/// BT.601 full-range YCbCr → RGB (the JFIF default conversion).
fn ycbcr_to_rgb(y: f32, cb: f32, cr: f32) -> [u8; 3] {
    let cb = cb - 128.0;
    let cr = cr - 128.0;
    let r = y + 1.402 * cr;
    let g = y - 0.344_136 * cb - 0.714_136 * cr;
    let b = y + 1.772 * cb;
    [
        r.round().clamp(0.0, 255.0) as u8,
        g.round().clamp(0.0, 255.0) as u8,
        b.round().clamp(0.0, 255.0) as u8,
    ]
}

/// Decode a §4a-bound AMV video frame to an upright RGB [`DecodedFrame`].
///
/// Performs the full baseline-JPEG decode of the bare entropy window
/// using the §4a hardcoded device profile (Annex K tables, 4:2:0,
/// baseline single scan), upsamples chroma by nearest-neighbour (2×),
/// converts YCbCr→RGB (BT.601 / JFIF), crops to the §2 `amvh` geometry,
/// and applies the §4a bottom-up vertical flip so the result is upright.
pub fn decode_frame(frame: &AmvVideoFrame<'_>) -> Result<DecodedFrame, AmvDemuxerError> {
    let width = frame.width() as usize;
    let height = frame.height() as usize;
    let (yp, cbp, crp) = decode_planes(frame.entropy_coded(), width, height)?;

    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let yv = yp.samples[y * yp.width + x] as f32;
            // 4:2:0: chroma sampled at half resolution, nearest-neighbour
            // upsample.
            let cx = x / 2;
            let cy = y / 2;
            let cb = cbp.samples[cy * cbp.width + cx] as f32;
            let cr = crp.samples[cy * crp.width + cx] as f32;
            let [r, g, b] = ycbcr_to_rgb(yv, cb, cr);
            let k = (y * width + x) * 3;
            rgb[k] = r;
            rgb[k + 1] = g;
            rgb[k + 2] = b;
        }
    }

    // §4a bottom-up orientation: the decoded raster is vertically
    // mirrored; a single flip yields the upright natural image.
    flip_rows_vertical(&mut rgb, height, width * 3);

    Ok(DecodedFrame {
        width: width as u32,
        height: height as u32,
        rgb,
    })
}

/// Convenience: bind §2 `header` geometry to a raw `00dc` `payload`
/// (strict §4a no-internal-markers check) and decode it to RGB in one
/// step.
pub fn decode_frame_from_payload(
    header: &AmvHeader,
    payload: &[u8],
) -> Result<DecodedFrame, AmvDemuxerError> {
    let frame = AmvVideoFrame::bind_strict(header, payload)?;
    decode_frame(&frame)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn huff_table_decodes_canonical_dc_luma() {
        // K.3 luma DC: BITS = [0,1,5,1,...]; the first length-2 code is
        // 00 → value 0, then 010..110 (length 3) → 1..5.
        let t = HuffTable::build(&DC_LUMA_BITS, &DC_LUMA_VALS);
        // Code "00" (2 bits) decodes to HUFFVAL[0] = 0.
        let mut br = BitReader::new(&[0b0000_0000]);
        assert_eq!(br.decode_huff(&t), 0);
        // Length-2 used the single code 0b00, so the length-3 run begins
        // at numeric code 0b010. "010" → first length-3 HUFFVAL = value 1;
        // "100" (= 0b100, the third length-3 code) → value 3.
        let mut br = BitReader::new(&[0b0100_0000]);
        assert_eq!(br.decode_huff(&t), 1);
        let mut br = BitReader::new(&[0b1000_0000]);
        assert_eq!(br.decode_huff(&t), 3);
    }

    #[test]
    fn receive_extend_sign_rules() {
        // size=3, bits "100" = 4 → positive (high bit set).
        let mut br = BitReader::new(&[0b1000_0000]);
        assert_eq!(br.receive_extend(3), 4);
        // size=3, bits "011" = 3 → negative branch: 3 - 8 + 1 = -4.
        let mut br = BitReader::new(&[0b0110_0000]);
        assert_eq!(br.receive_extend(3), -4);
        // size=0 → 0, consumes nothing.
        let mut br = BitReader::new(&[0xFF, 0x00]);
        assert_eq!(br.receive_extend(0), 0);
    }

    #[test]
    fn bitreader_resolves_byte_stuffing_and_marker_eof() {
        // FF 00 is a stuffed literal FF; FF D9 (EOI) ends entropy data.
        let mut br = BitReader::new(&[0xFF, 0x00, 0xFF, 0xD9]);
        // First 8 bits = 0xFF (the stuffed literal).
        assert_eq!(br.read_bits(8), 0xFF);
        // Next read hits the FF D9 marker → end of data, zero-padded.
        assert_eq!(br.read_bits(8), 0);
        assert!(br.exhausted);
    }

    #[test]
    fn idct_of_dc_only_block_is_flat() {
        // A block with only a DC term decodes to a constant tile equal to
        // DC/8 (the 2-D IDCT of a pure DC coefficient).
        let mut block = [0f32; 64];
        block[0] = 8.0 * 16.0; // arbitrary DC
        idct_8x8(&mut block);
        let first = block[0];
        for &v in block.iter() {
            assert!((v - first).abs() < 1e-3, "DC-only IDCT must be flat");
        }
        // Mean value equals DC / 8.
        assert!((first - 16.0).abs() < 1e-3);
    }

    #[test]
    fn decode_rejects_zero_geometry() {
        let mut h = comedian_header();
        h.width = 0;
        let payload = [0xFF, 0xD8, 0xFF, 0xD9];
        assert!(decode_frame_from_payload(&h, &payload).is_err());
    }

    #[test]
    fn decode_empty_scan_yields_mid_gray_frame() {
        // An empty entropy window (SOI immediately followed by EOI) leaves
        // every coefficient zero → every component decodes to the 128
        // level shift → a uniform mid-gray raster of the right geometry.
        let payload = [0xFF, 0xD8, 0xFF, 0xD9];
        let frame = decode_frame_from_payload(&comedian_header(), &payload)
            .expect("empty scan decodes to a flat frame");
        assert_eq!((frame.width, frame.height), (128, 96));
        assert_eq!(frame.rgb.len(), 128 * 96 * 3);
        // Y=128, Cb=Cr=128 → R=G=B=128.
        assert!(frame.rgb.iter().all(|&b| b == 128));
    }
}
