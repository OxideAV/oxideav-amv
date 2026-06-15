//! Typed video-frame surface for the `00dc` payload — the §4a
//! geometry-binding step of `docs/container/amv/amv-container-trace.md`.
//!
//! Per trace §4a ("Video chunk (`00dc`) — bare intra JPEG") an AMV
//! video chunk is a self-contained JPEG bracketed by `FF D8` (SOI) and
//! `FF D9` (EOI) with **no** internal marker segments: no `DQT`, no
//! `DHT`, no `SOF0`, no `SOS`. The quantization / Huffman tables and
//! the scan parameters are stripped from the bitstream and hardcoded
//! in the player's decoder, and — critically — *"the resolution comes
//! from `amvh` (§2)"*. A future AMV video decoder therefore cannot
//! consume a `00dc` payload in isolation: it needs the §2 stream
//! geometry bound to the frame bytes.
//!
//! [`AmvVideoFrame`] is exactly that binding. It is the typed,
//! validated input surface for the eventual decoder: §2 `width` ×
//! `height` from the parsed [`AmvHeader`] plus the §4a-validated
//! payload, with the entropy-coded window (the bytes strictly between
//! SOI and EOI) exposed as a slice. The decode step itself is **not**
//! implemented here — the trace documents the wire shape but is silent
//! on the hardcoded table contents, so anything past this structural
//! binding is not yet groundable.

use crate::parse::{
    validate_video_payload_no_internal_markers, validate_video_payload_shape, AmvHeader, JPEG_EOI,
    JPEG_SOI,
};
use crate::AmvDemuxerError;

/// A single AMV video frame: §2 stream geometry bound to a
/// §4a-validated `00dc` payload.
///
/// Constructed via [`AmvVideoFrame::bind`] (SOI / EOI bracket check)
/// or [`AmvVideoFrame::bind_strict`] (bracket check plus the §4a
/// no-internal-markers scan). Both constructors take the parsed
/// [`AmvHeader`] because the payload itself carries no geometry — per
/// trace §4a the `SOF`-style frame parameters are absent from the
/// bitstream and the resolution lives only in the §2 `amvh` header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AmvVideoFrame<'a> {
    /// Frame width in pixels, copied from [`AmvHeader::width`] (§2,
    /// `amvh` body offset +0x20).
    width: u32,
    /// Frame height in pixels, copied from [`AmvHeader::height`] (§2,
    /// `amvh` body offset +0x24).
    height: u32,
    /// The full validated `00dc` payload: `FF D8` + entropy-coded data
    /// + `FF D9` (§4a).
    body: &'a [u8],
}

impl<'a> AmvVideoFrame<'a> {
    /// Bind §2 geometry to a `00dc` payload after validating the §4a
    /// SOI / EOI bracket shape.
    ///
    /// Validation performed:
    ///
    /// * `header.width` and `header.height` are non-zero — §2 records
    ///   real pixel geometry for every observed device profile
    ///   (128 × 96 and 96 × 64), and a decoder cannot size its output
    ///   plane from a zero dimension.
    /// * [`validate_video_payload_shape`] on `body` — SOI at offset
    ///   `0`, EOI at `len - 2`, minimum 4 bytes (§4a).
    ///
    /// The §4a no-internal-markers invariant is **not** scanned here;
    /// callers wanting the full strict check use [`Self::bind_strict`].
    pub fn bind(header: &AmvHeader, body: &'a [u8]) -> Result<Self, AmvDemuxerError> {
        if header.width == 0 || header.height == 0 {
            return Err(AmvDemuxerError::InvalidData(format!(
                "video frame geometry must be non-zero, got {}x{} \
                 (trace §4a: resolution comes from the §2 amvh header)",
                header.width, header.height
            )));
        }
        validate_video_payload_shape(body)?;
        Ok(Self {
            width: header.width,
            height: header.height,
            body,
        })
    }

    /// Bind like [`Self::bind`], additionally enforcing the strict §4a
    /// no-internal-markers invariant via
    /// [`validate_video_payload_no_internal_markers`] — the trace's
    /// marker scan of the first frame found *only* SOI and EOI, so a
    /// payload carrying any `DQT` / `DHT` / `SOF` / `SOS` segment is
    /// not the table-stripped device-profile variant this crate
    /// documents.
    pub fn bind_strict(header: &AmvHeader, body: &'a [u8]) -> Result<Self, AmvDemuxerError> {
        let frame = Self::bind(header, body)?;
        validate_video_payload_no_internal_markers(body)?;
        Ok(frame)
    }

    /// Frame width in pixels (§2 `amvh` body offset +0x20).
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in pixels (§2 `amvh` body offset +0x24).
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The full validated `00dc` payload including the SOI / EOI
    /// bracket.
    pub fn body(&self) -> &'a [u8] {
        self.body
    }

    /// The entropy-coded window: the payload bytes strictly between
    /// the 2-byte SOI and the 2-byte EOI (§4a — "immediately after SOI
    /// the bytes are entropy-coded data"). Empty for the degenerate
    /// minimum 4-byte `SOI + EOI` payload.
    ///
    /// For the comedian fixture's first frame (1633-byte payload) this
    /// window is `1633 - 4 = 1629` bytes starting `E6 49 A6 93 …`
    /// (the §4a hexdump).
    pub fn entropy_coded(&self) -> &'a [u8] {
        &self.body[JPEG_SOI.len()..self.body.len() - JPEG_EOI.len()]
    }

    /// Whether this frame is a keyframe. Always `true`: per trace §4a
    /// the codec is intra-only — "Every frame is a keyframe".
    pub fn is_keyframe(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §2-shaped header carrying the comedian device profile
    /// (128 × 96 @ 12 fps, 1:33 packed duration).
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
    fn bind_minimal_soi_eoi_payload() {
        let body = [0xFF, 0xD8, 0xFF, 0xD9];
        let frame = AmvVideoFrame::bind(&comedian_header(), &body).expect("minimal bracket binds");
        assert_eq!(frame.width(), 128);
        assert_eq!(frame.height(), 96);
        assert_eq!(frame.body(), &body);
        assert!(frame.entropy_coded().is_empty());
        assert!(frame.is_keyframe());
    }

    #[test]
    fn bind_exposes_entropy_window_between_soi_and_eoi() {
        // §4a worked-example first bytes after SOI: E6 49 A6 93.
        let body = [0xFF, 0xD8, 0xE6, 0x49, 0xA6, 0x93, 0xFF, 0xD9];
        let frame = AmvVideoFrame::bind(&comedian_header(), &body).expect("binds");
        assert_eq!(frame.entropy_coded(), &[0xE6, 0x49, 0xA6, 0x93]);
    }

    #[test]
    fn bind_rejects_missing_soi() {
        let body = [0x00, 0xD8, 0xFF, 0xD9];
        assert!(AmvVideoFrame::bind(&comedian_header(), &body).is_err());
    }

    #[test]
    fn bind_rejects_missing_eoi() {
        let body = [0xFF, 0xD8, 0xFF, 0x00];
        assert!(AmvVideoFrame::bind(&comedian_header(), &body).is_err());
    }

    #[test]
    fn bind_rejects_sub_minimum_payload() {
        assert!(AmvVideoFrame::bind(&comedian_header(), &[0xFF, 0xD8, 0xD9]).is_err());
    }

    #[test]
    fn bind_rejects_zero_width() {
        let mut h = comedian_header();
        h.width = 0;
        assert!(AmvVideoFrame::bind(&h, &[0xFF, 0xD8, 0xFF, 0xD9]).is_err());
    }

    #[test]
    fn bind_rejects_zero_height() {
        let mut h = comedian_header();
        h.height = 0;
        assert!(AmvVideoFrame::bind(&h, &[0xFF, 0xD8, 0xFF, 0xD9]).is_err());
    }

    #[test]
    fn bind_strict_rejects_internal_marker_segment() {
        // A DQT marker (FF DB) inside the entropy window violates the
        // §4a no-internal-markers invariant; plain bind accepts the
        // bracket, strict bind must reject.
        let body = [0xFF, 0xD8, 0xFF, 0xDB, 0x12, 0xFF, 0xD9];
        assert!(AmvVideoFrame::bind(&comedian_header(), &body).is_ok());
        assert!(AmvVideoFrame::bind_strict(&comedian_header(), &body).is_err());
    }

    #[test]
    fn bind_strict_accepts_byte_stuffing() {
        // FF 00 inside the window is byte stuffing, not a marker.
        let body = [0xFF, 0xD8, 0xE6, 0xFF, 0x00, 0x49, 0xFF, 0xD9];
        let frame =
            AmvVideoFrame::bind_strict(&comedian_header(), &body).expect("stuffed FF accepted");
        assert_eq!(frame.entropy_coded(), &[0xE6, 0xFF, 0x00, 0x49]);
    }

    /// Real-fixture pin of the full §4a binding surface: parse the §2
    /// `amvh` header straight from the staged `comedian.amv` bytes,
    /// walk the `movi` payload, and strict-bind every `00dc` chunk.
    ///
    /// Pinned worked-example facts from the trace:
    ///
    /// * 1116 video frames, every one binding under the strict §4a
    ///   no-internal-markers scan, all 128 × 96, all keyframes.
    /// * First three frame payload sizes 1633 / 1627 / 1625 (§4 chunk
    ///   table at offsets 0x013C / 0x0B4C / 0x1556).
    /// * Frame 0's entropy window is `1633 - 4 = 1629` bytes and
    ///   begins `E6 49 A6 93` (§4a hexdump `FF D8 E6 49 A6 93 …`).
    #[test]
    fn comedian_fixture_strict_binds_all_1116_video_frames() {
        use crate::parse::{MoviPayload, MoviPayloadIter, AMVH_BODY_LEN, AMV_END_TRAILER};

        let crate_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/comedian.amv");
        let workspace_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/container/amv/fixtures/comedian.amv");
        let path = if crate_path.exists() {
            crate_path
        } else if workspace_path.exists() {
            workspace_path
        } else {
            eprintln!("skipping comedian AmvVideoFrame fixture test: not staged");
            return;
        };
        let bytes = std::fs::read(&path).expect("read fixture");

        // §2: the amvh body lives at file offset 0x20, 0x38 bytes.
        let header = AmvHeader::parse(&bytes[0x20..0x20 + AMVH_BODY_LEN as usize])
            .expect("comedian amvh parses");
        assert_eq!(header.width, 128);
        assert_eq!(header.height, 96);

        // §4: the movi body starts after the `movi` FOURCC and ends at
        // the §4c `AMV_END_` trailer.
        let movi_pos = bytes
            .windows(4)
            .position(|w| w == b"movi")
            .expect("movi FOURCC present");
        let trailer_start = bytes.len() - AMV_END_TRAILER.len();
        assert_eq!(&bytes[trailer_start..], &AMV_END_TRAILER);
        let movi_body = &bytes[movi_pos + 4..trailer_start];

        let mut n_frames = 0u32;
        let mut first_three_sizes = Vec::new();
        let mut first_entropy_head = None;
        let mut first_entropy_len = None;
        for item in MoviPayloadIter::new(movi_body) {
            let payload = item.expect("comedian.amv movi walk must not error");
            let MoviPayload::Video { body, .. } = payload else {
                continue;
            };
            let frame = AmvVideoFrame::bind_strict(&header, body)
                .expect("every comedian 00dc payload strict-binds under §4a");
            assert_eq!(frame.width(), 128);
            assert_eq!(frame.height(), 96);
            assert!(frame.is_keyframe());
            if first_three_sizes.len() < 3 {
                first_three_sizes.push(frame.body().len());
            }
            if first_entropy_head.is_none() {
                first_entropy_head = Some(frame.entropy_coded()[0..4].to_vec());
                first_entropy_len = Some(frame.entropy_coded().len());
            }
            n_frames += 1;
        }

        assert_eq!(n_frames, 1116, "expected 1116 bound video frames");
        assert_eq!(
            first_three_sizes,
            vec![1633, 1627, 1625],
            "§4 chunk-table sizes"
        );
        assert_eq!(
            first_entropy_head,
            Some(vec![0xE6, 0x49, 0xA6, 0x93]),
            "§4a first entropy-coded bytes after SOI"
        );
        assert_eq!(first_entropy_len, Some(1633 - 4), "entropy window length");
    }
}
