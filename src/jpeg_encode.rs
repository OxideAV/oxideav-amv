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

use crate::jpeg_decode::{dct_cos_table, DecodedFrame};
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

/// Where the stuffed entropy bytes of a [`BitWriter`] go: either a real
/// byte buffer ([`VecSink`], the materializing encode) or a bare byte
/// counter ([`CountSink`], the exact-size probe the rate-control budget
/// search runs — same stuffing, same final-pad behaviour, no
/// allocation).
trait EntropySink {
    fn emit(&mut self, b: u8);
}

/// Materializing sink: collects the stuffed entropy bytes.
struct VecSink(Vec<u8>);

impl EntropySink for VecSink {
    #[inline]
    fn emit(&mut self, b: u8) {
        self.0.push(b);
    }
}

/// Counting sink: tracks only how many stuffed bytes *would* be
/// emitted. Because stuffing decisions depend on the actual byte
/// values, the counter still sees every byte — the count is exact, not
/// an estimate.
struct CountSink(usize);

impl EntropySink for CountSink {
    #[inline]
    fn emit(&mut self, _b: u8) {
        self.0 += 1;
    }
}

/// MSB-first bit writer over the entropy-coded scan window. Re-applies
/// JPEG `FF`→`FF 00` byte stuffing so the emitted stream round-trips
/// through the decoder's de-stuffing [`crate::jpeg_decode`] `BitReader`.
struct BitWriter<S: EntropySink> {
    sink: S,
    acc: u32,
    nbits: u32,
}

impl<S: EntropySink> BitWriter<S> {
    fn with_sink(sink: S) -> Self {
        BitWriter {
            sink,
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
                self.sink.emit(b);
                if b == 0xFF {
                    self.sink.emit(0x00); // byte stuffing
                }
                self.nbits = 0;
                self.acc = 0;
            }
        }
    }

    /// Flush the final partial byte, padding the low bits with **1**s
    /// (T.81 §F.1.2.3: the trailing fill bits are 1s). Returns the sink.
    fn finish(mut self) -> S {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.acc = (self.acc << pad) | ((1u32 << pad) - 1);
            let b = (self.acc & 0xFF) as u8;
            self.sink.emit(b);
            if b == 0xFF {
                self.sink.emit(0x00);
            }
        }
        self.sink
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
    // Cosine basis from the decoder's shared precomputed table:
    // cos((2y+1)·v·π/16) = dct_cos_table()[v·8+y]. Entries are
    // bit-identical to the inline `cos()` calls this loop historically
    // made, so the encode output bytes are unchanged.
    let cos = dct_cos_table();
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
                s += block[y * 8 + x] * cos[v * 8 + y];
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
                s += tmp[v * 8 + x] * cos[u * 8 + x];
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

/// Natural-order index for each zig-zag position: `NATURAL_FROM_ZZ[k]`
/// is the natural 8×8 index whose zig-zag position (per [`ZIGZAG`]) is
/// `k`. Precomputed inverse of `ZIGZAG` so the per-coefficient entropy
/// walk does not re-scan the table for every coefficient.
const NATURAL_FROM_ZZ: [u8; 64] = {
    let mut out = [0u8; 64];
    let mut n = 0;
    while n < 64 {
        out[ZIGZAG[n] as usize] = n as u8;
        n += 1;
    }
    out
};

/// Quantize one natural-order coefficient block by `quant`
/// (round-to-nearest), producing the integer levels the fixed §4a
/// device dequant multiplies back out.
fn quantize_block(coeffs: &[f32; 64], quant: &[u8; 64]) -> [i32; 64] {
    let mut q = [0i32; 64];
    for (o, (&c, &qv)) in q.iter_mut().zip(coeffs.iter().zip(quant.iter())) {
        *o = (c / qv as f32).round() as i32;
    }
    q
}

/// Huffman-encode one already-quantized natural-order block (DC
/// difference + AC run/size) into `w`. `pred` carries the running DC
/// predictor for the component. Rate control does not tap into this
/// walk: a budgeted encode first *plans* a reduced block
/// ([`rd_optimize_block`]) and then encodes the planned levels through
/// this same unconditional path.
fn encode_block<S: EntropySink>(
    w: &mut BitWriter<S>,
    q: &[i32; 64],
    dc_tbl: &HuffEncTable,
    ac_tbl: &HuffEncTable,
    pred: &mut i32,
) {
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
    for &natural in NATURAL_FROM_ZZ.iter().skip(1) {
        // `natural`: the natural index at this zig-zag position.
        let coeff = q[natural as usize];
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
    let planes = prepare_planes_rgb(width, height, rgb)?;
    Ok(entropy_encode_blocks(
        &quantize_mcu_blocks(&planes),
        planes.mcus_x,
        planes.mcus_y,
    ))
}

/// MCU-pad-aligned float sample planes ready for the DCT stage — the
/// output of the RGB / native-YUV plane-fill front doors and the input
/// to [`quantize_mcu_blocks`]. Luma is `mcus_x·16 × mcus_y·16`, chroma
/// `mcus_x·8 × mcus_y·8` (4:2:0), all §4a bottom-up coded order with
/// edge replication into the pad.
struct PreparedPlanes {
    y_plane: Vec<f32>,
    luma_w: usize,
    mcus_x: usize,
    mcus_y: usize,
    cb_plane: Vec<f32>,
    cr_plane: Vec<f32>,
    chroma_w: usize,
}

/// RGB front door of the plane-fill stage (see [`encode_frame_rgb`] for
/// the profile description).
fn prepare_planes_rgb(
    width: u32,
    height: u32,
    rgb: &[u8],
) -> Result<PreparedPlanes, AmvDemuxerError> {
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

    Ok(PreparedPlanes {
        y_plane,
        luma_w,
        mcus_x,
        mcus_y,
        cb_plane,
        cr_plane,
        chroma_w,
    })
}

/// DCT + quantize the MCU grid of three prepared planes into the flat
/// quantized-block sequence the entropy stage consumes: 6 natural-order
/// blocks per MCU, MCU raster order, `[Y0, Y1, Y2, Y3, Cb, Cr]` within
/// each MCU (the §4a interleaved-scan order). Shared by
/// [`encode_frame_rgb`] and [`encode_frame_yuv420p`]: only the
/// plane-fill stage differs between the RGB and native-YUV front doors;
/// the DCT / quant walk is identical. Splitting quantization from the
/// entropy pass lets the budgeted encode re-run only the (cheap)
/// entropy stage per rate-search probe.
fn quantize_mcu_blocks(p: &PreparedPlanes) -> Vec<[i32; 64]> {
    let mut blocks = Vec::with_capacity(p.mcus_x * p.mcus_y * 6);
    for my in 0..p.mcus_y {
        for mx in 0..p.mcus_x {
            // Four luma blocks (raster order within the MCU).
            for by in 0..2usize {
                for bx in 0..2usize {
                    let ox = mx * 16 + bx * 8;
                    let oy = my * 16 + by * 8;
                    let mut blk = gather_block(&p.y_plane, p.luma_w, ox, oy);
                    fdct_8x8(&mut blk);
                    blocks.push(quantize_block(&blk, &QUANT_LUMA));
                }
            }
            // One Cb, one Cr block.
            let cox = mx * 8;
            let coy = my * 8;
            let mut cb_blk = gather_block(&p.cb_plane, p.chroma_w, cox, coy);
            fdct_8x8(&mut cb_blk);
            blocks.push(quantize_block(&cb_blk, &QUANT_CHROMA));
            let mut cr_blk = gather_block(&p.cr_plane, p.chroma_w, cox, coy);
            fdct_8x8(&mut cr_blk);
            blocks.push(quantize_block(&cr_blk, &QUANT_CHROMA));
        }
    }
    blocks
}

/// The four Annex K Huffman encode tables, built once and shared by
/// every entropy walk / RD plan of a frame encode (the budget search
/// runs many walks per frame).
struct EncTables {
    dc_luma: HuffEncTable,
    ac_luma: HuffEncTable,
    dc_chroma: HuffEncTable,
    ac_chroma: HuffEncTable,
}

impl EncTables {
    fn build() -> Self {
        EncTables {
            dc_luma: HuffEncTable::build(&DC_LUMA_BITS, &DC_LUMA_VALS),
            ac_luma: HuffEncTable::build(&AC_LUMA_BITS, &AC_LUMA_VALS),
            dc_chroma: HuffEncTable::build(&DC_CHROMA_BITS, &DC_CHROMA_VALS),
            ac_chroma: HuffEncTable::build(&AC_CHROMA_BITS, &AC_CHROMA_VALS),
        }
    }
}

/// Drive the interleaved-MCU entropy walk into `sink`. Shared core of
/// the materializing encode ([`entropy_encode_blocks`]) and the
/// exact-size probe ([`entropy_encoded_size`]).
fn entropy_encode_into<S: EntropySink>(blocks: &[[i32; 64]], t: &EncTables, sink: S) -> S {
    let EncTables {
        dc_luma,
        ac_luma,
        dc_chroma,
        ac_chroma,
    } = t;

    let mut bw = BitWriter::with_sink(sink);
    let mut pred_y = 0i32;
    let mut pred_cb = 0i32;
    let mut pred_cr = 0i32;

    for mcu in blocks.chunks_exact(6) {
        for luma in &mcu[..4] {
            encode_block(&mut bw, luma, dc_luma, ac_luma, &mut pred_y);
        }
        encode_block(&mut bw, &mcu[4], dc_chroma, ac_chroma, &mut pred_cb);
        encode_block(&mut bw, &mcu[5], dc_chroma, ac_chroma, &mut pred_cr);
    }
    bw.finish()
}

fn entropy_encode_blocks(blocks: &[[i32; 64]], mcus_x: usize, mcus_y: usize) -> Vec<u8> {
    debug_assert_eq!(blocks.len(), mcus_x * mcus_y * 6);
    entropy_encode_blocks_with(blocks, &EncTables::build())
}

/// [`entropy_encode_blocks`] with caller-provided prebuilt tables (the
/// budget search materializes its winning plan without rebuilding
/// them).
fn entropy_encode_blocks_with(blocks: &[[i32; 64]], t: &EncTables) -> Vec<u8> {
    let entropy = entropy_encode_into(blocks, t, VecSink(Vec::new())).0;
    let mut payload = Vec::with_capacity(entropy.len() + 4);
    payload.extend_from_slice(&[0xFF, 0xD8]); // SOI
    payload.extend_from_slice(&entropy);
    payload.extend_from_slice(&[0xFF, 0xD9]); // EOI
    payload
}

/// Exact byte size of the payload [`entropy_encode_blocks`] would
/// produce for `blocks` — the full stuffed entropy walk (stuffing
/// depends on actual byte values, so the walk is identical) plus the 4
/// SOI/EOI marker bytes — without allocating the payload.
fn entropy_encoded_size(blocks: &[[i32; 64]], t: &EncTables) -> usize {
    entropy_encode_into(blocks, t, CountSink(0)).0 + 4
}

// ---------------------------------------------------------------------
// Rate-controlled (budgeted) frame encode.
// ---------------------------------------------------------------------

/// The DC-only floor plan: every AC coefficient dropped, the smallest
/// §4a encode of this content (the DC predictor chain always survives —
/// dropping DC would shift whole tiles, not save meaningful bits).
fn dc_only_blocks(blocks: &[[i32; 64]]) -> Vec<[i32; 64]> {
    blocks
        .iter()
        .map(|b| {
            let mut o = [0i32; 64];
            o[0] = b[0];
            o
        })
        .collect()
}

/// Lagrangian rate–distortion plan for one quantized block: for every
/// nonzero AC coefficient choose to **keep it at full level**, **step
/// its magnitude down** to a lower size-category boundary, or **drop
/// it**, minimizing `distortion + lambda · bits` exactly by dynamic
/// programming.
///
/// The §4a device profile pins the quant tables in the *player*, so the
/// only rate lever that keeps a payload decodable is choosing which
/// quantized coefficients to spend bits on — and at what precision. For
/// a given price-per-bit `lambda` this planner solves that choice
/// optimally per block:
///
/// * **Distortion** of coding a coefficient of quantized level `q` at
///   reduced magnitude `m` (0 for a drop) under quantizer `Q` is
///   `((|q|−m)·Q)²` — the squared dequantized error. The 8×8 DCT is
///   orthogonal (up to one fixed scale factor shared by every
///   coefficient), so summed squared coefficient error *is* summed
///   squared pixel error: true MSE weighting, not a heuristic
///   frequency ramp.
/// * **Rate** is the exact Annex K entropy cost, including the run/size
///   coupling a coefficient's presence creates: dropping one lengthens
///   the zero run of the next kept one (possibly across a ZRL
///   boundary), the last kept coefficient decides whether an EOB is
///   emitted, and a stepped-down magnitude pays the (usually shorter)
///   run/size code and appended bits of its smaller category. The DP
///   walks kept-coefficient pairs, so every ZRL/EOB/size interaction is
///   priced exactly (only the whole-byte packing / stuffing granularity
///   is outside the model).
///
/// The step-down candidates for a level of magnitude category `s` are
/// the category boundaries `2^{s'} − 1` for `s' < s` (the largest
/// magnitude codable in `s'` bits) with the sign preserved — the
/// candidate set every intermediate magnitude is dominated by: any `m`
/// strictly inside a category costs that category's bits but more
/// distortion than its boundary.
///
/// The DP is over the (≤ 63) nonzero AC coefficients in zig-zag order:
/// `dp[i]` = the cheapest plan for the block prefix in which `i` is the
/// last kept coefficient (at its best candidate magnitude for the
/// arriving run), minimized over the previous kept coefficient (or
/// none). Backtracking materializes the plan into a copy of the block
/// with dropped levels zeroed and stepped-down levels replaced; DC
/// always survives untouched. Larger `lambda` monotonically favours
/// spending fewer bits, which is what makes the budget bisection in
/// [`entropy_encode_with_budget`] behave.
fn rd_optimize_block(q: &[i32; 64], quant: &[u8; 64], ac: &HuffEncTable, lambda: f64) -> [i32; 64] {
    // Collect the nonzero AC coefficients in zig-zag order.
    let mut zz_pos = [0u8; 63];
    let mut nat = [0u8; 63];
    let mut lvl = [0i32; 63];
    let mut n = 0usize;
    for (zz, &natural) in NATURAL_FROM_ZZ.iter().enumerate().skip(1) {
        let c = q[natural as usize];
        if c != 0 {
            zz_pos[n] = zz as u8;
            nat[n] = natural;
            lvl[n] = c;
            n += 1;
        }
    }
    if n == 0 {
        return *q;
    }
    let eob = ac.len[0x00] as f64;

    // Prefix sums of drop distortion: pre[i] = Σ dist(nz[0..i]).
    let mut pre = [0f64; 64];
    for i in 0..n {
        let d = lvl[i].unsigned_abs() as f64 * quant[nat[i] as usize] as f64;
        pre[i + 1] = pre[i] + d * d;
    }

    // Candidate magnitudes per nonzero: the full level plus each lower
    // size-category boundary, with the residual distortion of coding at
    // that magnitude. `cand_mag[i][c]` pairs with `cand_dist[i][c]` and
    // `cand_size[i][c]`; `cand_n[i]` counts them.
    const MAX_CAND: usize = 11;
    let mut cand_mag = [[0u32; MAX_CAND]; 63];
    let mut cand_size = [[0u32; MAX_CAND]; 63];
    let mut cand_dist = [[0f64; MAX_CAND]; 63];
    let mut cand_n = [0usize; 63];
    for i in 0..n {
        let mag = lvl[i].unsigned_abs();
        let size = magnitude_category(lvl[i]).0;
        let qv = quant[nat[i] as usize] as f64;
        cand_mag[i][0] = mag;
        cand_size[i][0] = size;
        cand_dist[i][0] = 0.0;
        let mut c = 1usize;
        for s in (1..size).rev() {
            let m = (1u32 << s) - 1;
            let err = (mag - m) as f64 * qv;
            cand_mag[i][c] = m;
            cand_size[i][c] = s;
            cand_dist[i][c] = err * err;
            c += 1;
        }
        cand_n[i] = c;
    }

    // Cheapest way to code nonzero `i` after `run` zeros: best candidate
    // magnitude for that arriving run. Returns (cost, magnitude).
    let code_after_run = |i: usize, run: usize| -> (f64, u32) {
        let zrl_cost = lambda * ((run / 16) as u32 * ac.len[0xF0] as u32) as f64;
        let rem = run % 16;
        let mut best = f64::INFINITY;
        let mut best_mag = cand_mag[i][0];
        for c in 0..cand_n[i] {
            let size = cand_size[i][c] as usize;
            let bits = (ac.len[(rem << 4) | size] as usize + size) as f64;
            let cost = cand_dist[i][c] + lambda * bits;
            if cost < best {
                best = cost;
                best_mag = cand_mag[i][c];
            }
        }
        (zrl_cost + best, best_mag)
    };

    // dp[i]: cheapest cost over zig-zag positions 1..=zz(i) with nz `i`
    // kept last; prev[i] backtracks the kept chain (-1 = none before);
    // mag[i] the magnitude `i` is coded at on its best path.
    let mut dp = [0f64; 63];
    let mut prev = [-1i8; 63];
    let mut mag = [0u32; 63];
    for i in 0..n {
        let zi = zz_pos[i] as usize;
        // Previous kept = none: every nonzero before `i` is dropped.
        let (cost, m) = code_after_run(i, zi - 1);
        let mut best = pre[i] + cost;
        let mut best_j = -1i8;
        let mut best_m = m;
        for j in 0..i {
            let run = zi - zz_pos[j] as usize - 1;
            let (cost, m) = code_after_run(i, run);
            let c = dp[j] + (pre[i] - pre[j + 1]) + cost;
            if c < best {
                best = c;
                best_j = j as i8;
                best_m = m;
            }
        }
        dp[i] = best;
        prev[i] = best_j;
        mag[i] = best_m;
    }

    // Close the block: either every nonzero is dropped (DC + EOB), or
    // the plan ends at kept `i` (EOB unless `i` sits at position 63).
    let mut best = pre[n] + lambda * eob;
    let mut best_end = -1i8;
    for i in 0..n {
        let tail_eob = if (zz_pos[i] as usize) < 63 { eob } else { 0.0 };
        let c = dp[i] + (pre[n] - pre[i + 1]) + lambda * tail_eob;
        if c < best {
            best = c;
            best_end = i as i8;
        }
    }

    // Materialize the plan.
    let mut out = [0i32; 64];
    out[0] = q[0];
    let mut i = best_end;
    while i >= 0 {
        let k = i as usize;
        out[nat[k] as usize] = if lvl[k] < 0 {
            -(mag[k] as i32)
        } else {
            mag[k] as i32
        };
        i = prev[k];
    }
    out
}

/// Run [`rd_optimize_block`] over every block of the frame at one
/// `lambda`, writing the planned blocks into the reusable `out` buffer.
/// Block interleave per MCU is 4 luma + Cb + Cr (§4a 4:2:0), so blocks
/// `idx % 6 < 4` price against the luma tables and the rest against the
/// chroma tables.
fn rd_plan_blocks(blocks: &[[i32; 64]], t: &EncTables, lambda: f64, out: &mut Vec<[i32; 64]>) {
    out.clear();
    out.extend(blocks.iter().enumerate().map(|(idx, b)| {
        if idx % 6 < 4 {
            rd_optimize_block(b, &QUANT_LUMA, &t.ac_luma, lambda)
        } else {
            rd_optimize_block(b, &QUANT_CHROMA, &t.ac_chroma, lambda)
        }
    }));
}

/// Result of a rate-controlled frame encode
/// ([`encode_frame_rgb_with_budget`] /
/// [`encode_frame_yuv420p_with_budget`]).
#[derive(Debug, Clone)]
pub struct BudgetedFrame {
    /// The bare `00dc` payload (`FF D8` + stuffed entropy + `FF D9`),
    /// always a fully conforming §4a frame regardless of how hard the
    /// budget squeezed — the fixed device tables decode it unchanged.
    pub payload: Vec<u8>,
    /// `true` when `payload.len() <= max_payload_bytes`. `false` only
    /// when the budget was below the frame's DC-only floor — the
    /// returned payload is then that floor (the smallest §4a encode of
    /// this content), so a stream-level rate controller can absorb the
    /// overshoot on later frames instead of failing the frame.
    pub within_budget: bool,
}

/// Bisect the Lagrangian price-per-bit `λ` to the cheapest
/// rate–distortion plan ([`rd_optimize_block`]) that fits
/// `max_payload_bytes`, over blocks quantized once up front.
///
/// A larger `λ` makes every per-block DP drop at least as much (the
/// classic Lagrangian sweep), so payload size is near-monotone in `λ`;
/// the search first brackets a fitting `λ` geometrically, then bisects,
/// keeping the invariant that the returned plan's **measured** size fit
/// (each probe is the exact counting walk [`entropy_encoded_size`] —
/// the rare monotonicity blip from byte-packing granularity can cost a
/// slightly-less-optimal `λ` pick but never a budget violation). Only
/// the winning plan is materialized to bytes.
fn entropy_encode_with_budget(
    blocks: &[[i32; 64]],
    mcus_x: usize,
    mcus_y: usize,
    max_payload_bytes: usize,
) -> BudgetedFrame {
    debug_assert_eq!(blocks.len(), mcus_x * mcus_y * 6);
    let tables = EncTables::build();
    if entropy_encoded_size(blocks, &tables) <= max_payload_bytes {
        return BudgetedFrame {
            payload: entropy_encode_blocks_with(blocks, &tables),
            within_budget: true,
        };
    }
    let floor = dc_only_blocks(blocks);
    if entropy_encoded_size(&floor, &tables) > max_payload_bytes {
        return BudgetedFrame {
            payload: entropy_encode_blocks_with(&floor, &tables),
            within_budget: false,
        };
    }

    // Bracket: grow λ geometrically until the plan fits. λ = distortion
    // per bit; the largest useful price is bounded by the largest
    // single-coefficient distortion (≤ (2048)² — the DCT magnitude
    // bound — over ≥ 2 bits), so the ×16 ladder from 1 reaches a
    // fitting λ within ~6 steps on any input; the DC-only floor (which
    // fits, checked above) is the safety net.
    let mut plan = Vec::with_capacity(blocks.len());
    let mut best: Option<Vec<[i32; 64]>> = None;
    let mut best_size = 0usize;
    let mut lo = 0.0f64;
    let mut hi = 1.0f64;
    for _ in 0..24 {
        rd_plan_blocks(blocks, &tables, hi, &mut plan);
        let size = entropy_encoded_size(&plan, &tables);
        if size <= max_payload_bytes {
            best = Some(plan.clone());
            best_size = size;
            break;
        }
        lo = hi;
        hi *= 16.0;
    }
    let mut best = best.unwrap_or(floor);

    // Bisect λ down to the lightest plan that still fits, stopping
    // early once the kept plan uses ≥ 99.5 % of the budget (further λ
    // resolution cannot buy a visible quality step at that point).
    // Invariant: `hi` produced `best` (measured to fit), `lo` did not.
    for _ in 0..12 {
        if best_size * 1000 >= max_payload_bytes * 995 {
            break;
        }
        let mid = 0.5 * (lo + hi);
        rd_plan_blocks(blocks, &tables, mid, &mut plan);
        let size = entropy_encoded_size(&plan, &tables);
        if size <= max_payload_bytes {
            hi = mid;
            best_size = size;
            std::mem::swap(&mut best, &mut plan);
        } else {
            lo = mid;
        }
    }

    let payload = entropy_encode_blocks_with(&best, &tables);
    debug_assert!(payload.len() <= max_payload_bytes);
    BudgetedFrame {
        payload,
        within_budget: true,
    }
}

/// Rate-controlled variant of [`encode_frame_rgb`]: encode an upright
/// RGB raster into a bare `00dc` payload of **at most
/// `max_payload_bytes`** when that budget is achievable.
///
/// The §4a device profile hardcodes the quant / Huffman tables in the
/// player, so an AMV encoder cannot trade quality for bits by scaling
/// quantization the way a generic JPEG encoder would. The only lever
/// that keeps the payload decodable by the fixed tables is choosing
/// which quantized coefficients to spend bits on. This encode makes
/// that choice by exact per-block Lagrangian rate–distortion
/// optimization (`rd_optimize_block`): each block keeps the
/// coefficient subset minimizing `MSE + λ·bits` under the true Annex K
/// entropy cost (run/size codes, ZRL splits, EOB), and the price `λ` is
/// bisected to the lightest plan that fits the budget. The DCT +
/// quantization run once; each `λ` probe re-plans the blocks and runs
/// an allocation-free exact-size counting walk of the entropy stage,
/// and only the winning plan is materialized to bytes.
///
/// A budget at or above the unconstrained encode size returns the
/// byte-identical unconstrained payload. A budget below the frame's
/// DC-only floor returns that floor with
/// [`BudgetedFrame::within_budget`] `== false` rather than failing, so
/// stream-level controllers can carry the debt. Geometry / length
/// validation matches [`encode_frame_rgb`].
pub fn encode_frame_rgb_with_budget(
    width: u32,
    height: u32,
    rgb: &[u8],
    max_payload_bytes: usize,
) -> Result<BudgetedFrame, AmvDemuxerError> {
    let planes = prepare_planes_rgb(width, height, rgb)?;
    Ok(entropy_encode_with_budget(
        &quantize_mcu_blocks(&planes),
        planes.mcus_x,
        planes.mcus_y,
        max_payload_bytes,
    ))
}

/// Rate-controlled variant of [`encode_frame_yuv420p`] — the native
/// planar front door of [`encode_frame_rgb_with_budget`], with the same
/// budget semantics and the same plane-geometry validation as
/// [`encode_frame_yuv420p`].
pub fn encode_frame_yuv420p_with_budget(
    width: u32,
    height: u32,
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    max_payload_bytes: usize,
) -> Result<BudgetedFrame, AmvDemuxerError> {
    let planes = prepare_planes_yuv420p(width, height, y, cb, cr)?;
    Ok(entropy_encode_with_budget(
        &quantize_mcu_blocks(&planes),
        planes.mcus_x,
        planes.mcus_y,
        max_payload_bytes,
    ))
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
    let planes = prepare_planes_yuv420p(width, height, y, cb, cr)?;
    Ok(entropy_encode_blocks(
        &quantize_mcu_blocks(&planes),
        planes.mcus_x,
        planes.mcus_y,
    ))
}

/// Native-YUV420P front door of the plane-fill stage (see
/// [`encode_frame_yuv420p`] for the profile description).
fn prepare_planes_yuv420p(
    width: u32,
    height: u32,
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
) -> Result<PreparedPlanes, AmvDemuxerError> {
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

    Ok(PreparedPlanes {
        y_plane,
        luma_w,
        mcus_x,
        mcus_y,
        cb_plane,
        cr_plane,
        chroma_w,
    })
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
    fn natural_from_zz_is_the_inverse_of_zigzag() {
        // NATURAL_FROM_ZZ[k] must be the natural index whose ZIGZAG
        // entry is k — the precompute must match the linear scan the
        // encoder historically performed.
        for (zz, &natural) in NATURAL_FROM_ZZ.iter().enumerate() {
            let scanned = ZIGZAG.iter().position(|&z| z as usize == zz).unwrap();
            assert_eq!(natural as usize, scanned, "zz={zz}");
        }
    }

    /// Deterministic structured-noise RGB test frame: smooth gradients
    /// plus a pseudo-random texture so the unconstrained encode spends
    /// real AC bits (a flat frame would leave nothing to trim).
    fn textured_rgb(w: u32, h: u32) -> Vec<u8> {
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        let mut lcg: u32 = 0x1234_5678;
        for y in 0..h {
            for x in 0..w {
                lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (lcg >> 24) as i32 - 128;
                let k = ((y * w + x) * 3) as usize;
                let base = (x * 2 + y * 3) as i32;
                rgb[k] = (base % 200 + 28 + noise / 4).clamp(0, 255) as u8;
                rgb[k + 1] = ((base * 2) % 180 + 40 + noise / 6).clamp(0, 255) as u8;
                rgb[k + 2] = ((x + y) as i32 % 160 + 48 + noise / 8).clamp(0, 255) as u8;
            }
        }
        rgb
    }

    fn mae(a: &[u8], b: &[u8]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x as f64 - y as f64).abs())
            .sum::<f64>()
            / a.len() as f64
    }

    #[test]
    fn budgeted_encode_with_generous_budget_is_the_unconstrained_encode() {
        let (w, h) = (64u32, 48u32);
        let rgb = textured_rgb(w, h);
        let unconstrained = encode_frame_rgb(w, h, &rgb).expect("encode");
        let budgeted =
            encode_frame_rgb_with_budget(w, h, &rgb, unconstrained.len()).expect("budgeted encode");
        assert!(budgeted.within_budget);
        assert_eq!(
            budgeted.payload, unconstrained,
            "a budget equal to the unconstrained size must return the unconstrained bytes"
        );
    }

    #[test]
    fn counting_size_probe_equals_materialized_length() {
        // The budget search trusts the counting walk as an *exact* size
        // oracle (stuffing + final-pad included); pin it to the
        // materializing walk across RD plans, geometries and contents
        // (flat content exercises the EOB-heavy path, textured content
        // the stuffing path).
        let tables = EncTables::build();
        for (w, h) in [(64u32, 48u32), (17, 17), (16, 16)] {
            let planes_t = prepare_planes_rgb(w, h, &textured_rgb(w, h)).expect("planes");
            let flat = vec![200u8; (w * h * 3) as usize];
            let planes_f = prepare_planes_rgb(w, h, &flat).expect("planes");
            for planes in [&planes_t, &planes_f] {
                let blocks = quantize_mcu_blocks(planes);
                let mut candidates = vec![blocks.clone(), dc_only_blocks(&blocks)];
                for lambda in [0.5f64, 8.0, 200.0, 1e6] {
                    let mut plan = Vec::new();
                    rd_plan_blocks(&blocks, &tables, lambda, &mut plan);
                    candidates.push(plan);
                }
                for (ci, cand) in candidates.iter().enumerate() {
                    let bytes = entropy_encode_blocks(cand, planes.mcus_x, planes.mcus_y);
                    assert_eq!(
                        entropy_encoded_size(cand, &tables),
                        bytes.len(),
                        "counting probe diverged at {w}x{h} candidate {ci}"
                    );
                }
            }
        }
    }

    #[test]
    fn rd_plan_is_lambda_monotone_and_bounded_by_floor_and_full() {
        // Larger λ must never spend more bits, λ→0 approaches the full
        // encode, and a huge λ collapses to the DC-only floor.
        let (w, h) = (64u32, 48u32);
        let planes = prepare_planes_rgb(w, h, &textured_rgb(w, h)).expect("planes");
        let blocks = quantize_mcu_blocks(&planes);
        let tables = EncTables::build();

        let full = entropy_encoded_size(&blocks, &tables);
        let floor = entropy_encoded_size(&dc_only_blocks(&blocks), &tables);
        let mut last = usize::MAX;
        let mut plan = Vec::new();
        for lambda in [0.0f64, 1.0, 10.0, 100.0, 1e3, 1e5, 1e9] {
            rd_plan_blocks(&blocks, &tables, lambda, &mut plan);
            let size = entropy_encoded_size(&plan, &tables);
            assert!(size <= last, "λ={lambda}: size {size} grew past {last}");
            assert!((floor..=full).contains(&size));
            last = size;
        }
        // λ = 0 keeps everything; a huge λ is the DC-only floor.
        rd_plan_blocks(&blocks, &tables, 0.0, &mut plan);
        assert_eq!(plan, blocks);
        rd_plan_blocks(&blocks, &tables, 1e12, &mut plan);
        assert_eq!(plan, dc_only_blocks(&blocks));
    }

    #[test]
    fn rd_plan_only_attenuates_ac_coefficients_and_keeps_dc() {
        // The plan may only *attenuate* nonzero ACs — drop them to zero
        // or step their magnitude down (sign preserved) — never grow a
        // level, flip a sign, invent a coefficient, or touch DC. (The
        // §4a fixed tables decode whatever levels we emit; correctness
        // of the plan is that every planned payload stays a faithful,
        // lighter encode of the same frame.) A stepped-down magnitude
        // must sit exactly on a lower size-category boundary `2^s − 1`
        // (any other reduced magnitude is RD-dominated by a boundary).
        let (w, h) = (48u32, 48u32);
        let planes = prepare_planes_rgb(w, h, &textured_rgb(w, h)).expect("planes");
        let blocks = quantize_mcu_blocks(&planes);
        let tables = EncTables::build();
        let mut plan = Vec::new();
        let mut dropped = 0usize;
        let mut stepped = 0usize;
        for lambda in [5.0f64, 50.0, 500.0] {
            rd_plan_blocks(&blocks, &tables, lambda, &mut plan);
            for (b, p) in blocks.iter().zip(&plan) {
                assert_eq!(b[0], p[0], "DC must survive planning");
                for k in 1..64 {
                    if p[k] == b[k] {
                        continue;
                    }
                    assert_ne!(b[k], 0, "planning must not invent coefficient {k}");
                    if p[k] == 0 {
                        dropped += 1;
                        continue;
                    }
                    assert_eq!(
                        p[k].signum(),
                        b[k].signum(),
                        "step-down must preserve sign at {k}"
                    );
                    let (pm, bm) = (p[k].unsigned_abs(), b[k].unsigned_abs());
                    assert!(pm < bm, "planned |{pm}| must shrink from |{bm}| at {k}");
                    assert!(
                        (pm + 1).is_power_of_two(),
                        "stepped-down magnitude {pm} must be a size-category boundary"
                    );
                    stepped += 1;
                }
            }
        }
        assert!(dropped > 0, "a binding λ must drop coefficients");
        assert!(
            stepped > 0,
            "some λ on textured content must engage magnitude step-down"
        );
    }

    #[test]
    fn budgeted_encode_meets_shrinking_budgets_and_degrades_gracefully() {
        let (w, h) = (64u32, 48u32);
        let rgb = textured_rgb(w, h);
        let unconstrained = encode_frame_rgb(w, h, &rgb).expect("encode");
        let full_len = unconstrained.len();
        let reference =
            decode_frame_from_payload(&header_wh(w, h), &unconstrained).expect("decode full");
        let full_mae = mae(&rgb, &reference.rgb);

        let mut last_len = full_len + 1;
        for frac in [3usize, 4, 6, 10] {
            let budget = full_len * 2 / frac; // 2/3, 1/2, 1/3, 1/5
            let b = encode_frame_rgb_with_budget(w, h, &rgb, budget).expect("budgeted");
            assert!(b.within_budget, "budget {budget} should be achievable");
            assert!(
                b.payload.len() <= budget,
                "payload {} exceeds budget {budget}",
                b.payload.len()
            );
            assert!(
                b.payload.len() <= last_len,
                "smaller budget must not grow the payload"
            );
            last_len = b.payload.len();
            // Every budgeted payload stays a conforming §4a frame the
            // fixed device tables decode.
            let frame = crate::video::AmvVideoFrame::bind_strict(&header_wh(w, h), &b.payload)
                .expect("budgeted payload passes the strict §4a bind");
            assert_eq!((frame.width(), frame.height()), (w, h));
            let decoded = decode_frame_from_payload(&header_wh(w, h), &b.payload).expect("decode");
            let m = mae(&rgb, &decoded.rgb);
            assert!(
                m >= full_mae - 0.05,
                "trimming cannot beat the unconstrained encode: {m} < {full_mae}"
            );
            assert!(
                m < 40.0,
                "budget {budget} ({}/{} of full): degradation must stay graceful, MAE {m}",
                2,
                frac
            );
        }
    }

    #[test]
    fn budgeted_encode_below_floor_returns_floor_not_error() {
        let (w, h) = (32u32, 32u32);
        let rgb = textured_rgb(w, h);
        // 4 bytes can never hold SOI + any entropy + EOI.
        let b = encode_frame_rgb_with_budget(w, h, &rgb, 4).expect("floor encode");
        assert!(!b.within_budget, "4-byte budget is below the DC-only floor");
        assert!(b.payload.len() > 4);
        // The floor payload is still a valid, decodable §4a frame.
        let decoded =
            decode_frame_from_payload(&header_wh(w, h), &b.payload).expect("floor decodes");
        assert_eq!((decoded.width, decoded.height), (w, h));
        // And it is the DC-only encode: much smaller than unconstrained
        // on textured content.
        let full = encode_frame_rgb(w, h, &rgb).expect("full");
        assert!(
            b.payload.len() * 2 < full.len(),
            "DC-only floor {} should be far below the full encode {}",
            b.payload.len(),
            full.len()
        );
    }

    #[test]
    fn budgeted_encode_rejects_bad_geometry_like_the_unbudgeted_path() {
        assert!(encode_frame_rgb_with_budget(0, 8, &[], 100).is_err());
        assert!(encode_frame_rgb_with_budget(8, 8, &[0u8; 10], 100).is_err());
        assert!(
            encode_frame_yuv420p_with_budget(16, 16, &[0u8; 100], &[0u8; 64], &[0u8; 64], 100)
                .is_err()
        );
    }

    #[test]
    fn budgeted_yuv420p_matches_budgeted_rgb_on_shared_content() {
        // The two front doors share one quantized representation, so at
        // the same budget they must pick the same trim and produce the
        // same bytes when fed the §4a fixed-point planes.
        use crate::jpeg_decode::decode_frame_yuv420p_from_payload;
        let (w, h) = (48u32, 32u32);
        let rgb = textured_rgb(w, h);
        let full = encode_frame_rgb(w, h, &rgb).expect("full");
        let yuv = decode_frame_yuv420p_from_payload(&header_wh(w, h), &full).expect("yuv");
        let budget = full.len() / 2;
        let via_yuv = encode_frame_yuv420p_with_budget(w, h, &yuv.y, &yuv.cb, &yuv.cr, budget)
            .expect("yuv budgeted");
        assert!(via_yuv.within_budget);
        assert!(via_yuv.payload.len() <= budget);
        // Same planes through the unbudgeted YUV encode must reproduce
        // `full` (fixed point) — pin that the budget path diverges only
        // by trimming.
        let unbudgeted = encode_frame_yuv420p(w, h, &yuv.y, &yuv.cb, &yuv.cr).expect("yuv full");
        assert_eq!(unbudgeted, full);
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
