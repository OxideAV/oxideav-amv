//! `oxideav-core` [`Decoder`] / [`Encoder`] trait surface for the AMV
//! intrinsic video codec (`amv_video`).
//!
//! AMV's video is a **table-stripped** baseline JPEG (trace §4a): every
//! `00dc` payload is `FF D8` + bare entropy + `FF D9`, with the quant /
//! Huffman tables, frame geometry and scan parameters all absent from
//! the wire and hardcoded in the device. A generic `mjpeg` decoder
//! therefore **cannot** consume a `00dc` payload directly — it needs the
//! tables spliced back in first (the
//! [`reconstruct_jpeg`](crate::reconstruct_jpeg) path). This module is
//! the *direct* registry surface: it decodes / encodes the bare
//! bitstream straight via [`decode_frame_yuv420p`] /
//! [`encode_frame_yuv420p`] against the §4a device profile, under a
//! dedicated `amv_video` codec id so it never collides with the generic
//! `mjpeg` id the dedicated `oxideav-mjpeg` crate owns.
//!
//! # The `amv_video` profile (trace §4a)
//!
//! * **YUV420P**, BT.601 / JFIF, 8-bit, intra-only — every frame is a
//!   keyframe.
//! * Geometry comes from the stream's
//!   [`CodecParameters`](oxideav_core::CodecParameters) `width` /
//!   `height` (the §2 `amvh` resolution); the bitstream carries none.
//! * Decoder output is one [`VideoFrame`] with three planes (Y at
//!   `width × height`, Cb / Cr at `ceil(width/2) × ceil(height/2)`); the
//!   encoder is the byte-inverse, one bare `00dc` payload per frame.
//!
//! The decode / encode are stateless per frame (intra-only), so `reset`
//! just drains.

use std::collections::VecDeque;

use oxideav_core::{
    CodecId, CodecParameters, Decoder, Encoder, Error, Frame, MediaType, Packet, PixelFormat,
    Result, TimeBase, VideoFrame, VideoPlane,
};

use crate::jpeg_decode::decode_frame_yuv420p_from_payload;
use crate::jpeg_encode::encode_frame_yuv420p;
use crate::parse::AmvHeader;

/// Build the parse-level [`AmvHeader`] geometry binding the decode /
/// encode functions need, from a width/height pair. Only `width` /
/// `height` are read by the §4a JPEG path; the other fields are inert
/// placeholders.
fn header_from_dims(width: u32, height: u32) -> AmvHeader {
    AmvHeader {
        micros_per_frame: 0,
        width,
        height,
        fps: 0,
        flag_one: 1,
        reserved_30: 0,
        duration_packed: 0,
    }
}

/// Validate that a [`CodecParameters`] carries a usable non-zero §2
/// geometry, returning `(width, height)`.
fn dims_of(params: &CodecParameters) -> Result<(u32, u32)> {
    let w = params.width.unwrap_or(0);
    let h = params.height.unwrap_or(0);
    if w == 0 || h == 0 {
        return Err(Error::unsupported(
            "amv_video: stream width/height must be set (the §2 amvh geometry is not on the wire)",
        ));
    }
    Ok((w, h))
}

// ───────────────────────────── decoder ─────────────────────────────

/// Build a boxed [`Decoder`] for the AMV `amv_video` codec.
///
/// Direct-factory entry point — [`crate::register_codecs`] installs this
/// same function. `params.width` / `params.height` must carry the §2
/// `amvh` geometry (the bitstream has none); a missing dimension is
/// rejected at construction.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let (width, height) = dims_of(params)?;
    Ok(Box::new(AmvVideoDecoder {
        codec_id: params.codec_id.clone(),
        width,
        height,
        pending: None,
        eof: false,
    }))
}

/// AMV `amv_video` decoder: one bare `00dc` payload in, one YUV420P
/// [`VideoFrame`] out. Intra-only (every frame is a keyframe), so it
/// holds no cross-packet state and `reset` only drains.
#[derive(Debug)]
pub struct AmvVideoDecoder {
    codec_id: CodecId,
    width: u32,
    height: u32,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for AmvVideoDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "amv_video decoder: call receive_frame before sending another packet",
            ));
        }
        self.pending = Some(packet.clone());
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        let Some(pkt) = self.pending.take() else {
            return if self.eof {
                Err(Error::Eof)
            } else {
                Err(Error::NeedMore)
            };
        };
        let header = header_from_dims(self.width, self.height);
        let yuv = decode_frame_yuv420p_from_payload(&header, &pkt.data).map_err(Error::from)?;
        let frame = VideoFrame {
            pts: pkt.pts,
            planes: vec![
                VideoPlane {
                    stride: yuv.width as usize,
                    data: yuv.y,
                },
                VideoPlane {
                    stride: yuv.chroma_width as usize,
                    data: yuv.cb,
                },
                VideoPlane {
                    stride: yuv.chroma_width as usize,
                    data: yuv.cr,
                },
            ],
        };
        Ok(Frame::Video(frame))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.pending = None;
        self.eof = false;
        Ok(())
    }
}

// ───────────────────────────── encoder ─────────────────────────────

/// Build a boxed [`Encoder`] for the AMV `amv_video` codec.
///
/// Direct-factory counterpart to [`make_decoder`]. `params.width` /
/// `params.height` set the §2 geometry every input frame must match.
/// [`Encoder::output_params`] declare YUV420P at that geometry so a
/// downstream [`crate::AmvMuxer`] sizes the `amvh` header correctly.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let (width, height) = dims_of(params)?;
    let mut output = params.clone();
    output.media_type = MediaType::Video;
    output.codec_id = params.codec_id.clone();
    output.width = Some(width);
    output.height = Some(height);
    output.pixel_format = Some(PixelFormat::Yuv420P);

    // Packet time base is the frame interval `1/fps` when the caller set
    // a frame rate (the AMV stream clock the demuxer/muxer use), else a
    // neutral `1/1` so a frame's pts passes through as a raw frame index.
    let time_base = match params.frame_rate {
        Some(r) if r.num > 0 && r.den > 0 => TimeBase::new(r.den, r.num),
        _ => TimeBase::new(1, 1),
    };

    Ok(Box::new(AmvVideoEncoder {
        output,
        width,
        height,
        time_base,
        queue: VecDeque::new(),
    }))
}

/// AMV `amv_video` encoder: one YUV420P [`VideoFrame`] in, one bare
/// `00dc` payload out.
#[derive(Debug)]
pub struct AmvVideoEncoder {
    output: CodecParameters,
    width: u32,
    height: u32,
    time_base: TimeBase,
    queue: VecDeque<Packet>,
}

impl Encoder for AmvVideoEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let Frame::Video(v) = frame else {
            return Err(Error::invalid("amv_video encoder: video frames only"));
        };
        if v.planes.len() != 3 {
            return Err(Error::invalid(format!(
                "amv_video encoder: expected 3 YUV420P planes, got {}",
                v.planes.len()
            )));
        }
        let w = self.width as usize;
        let h = self.height as usize;
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);

        // De-stride each plane into the tight buffer the §4a encoder
        // expects (luma w×h, chroma cw×ch). A stride wider than the row
        // width carries trailing padding bytes that must be dropped.
        let y = pack_plane(&v.planes[0], w, h)?;
        let cb = pack_plane(&v.planes[1], cw, ch)?;
        let cr = pack_plane(&v.planes[2], cw, ch)?;

        let payload =
            encode_frame_yuv420p(self.width, self.height, &y, &cb, &cr).map_err(Error::from)?;
        let mut pkt = Packet::new(0, self.time_base, payload);
        pkt.pts = v.pts;
        pkt.dts = v.pts;
        pkt.flags.keyframe = true; // intra-only: every frame is a keyframe
        self.queue.push_back(pkt);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        self.queue.pop_front().ok_or(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Copy the `row_w × rows` tight window out of a (possibly padded)
/// [`VideoPlane`] whose row stride is `plane.stride` (≥ `row_w`).
fn pack_plane(plane: &VideoPlane, row_w: usize, rows: usize) -> Result<Vec<u8>> {
    let stride = plane.stride;
    if stride < row_w {
        return Err(Error::invalid(format!(
            "amv_video encoder: plane stride {stride} < row width {row_w}"
        )));
    }
    if plane.data.len() < stride * rows {
        return Err(Error::invalid(format!(
            "amv_video encoder: plane data {} too short for {rows} rows of stride {stride}",
            plane.data.len()
        )));
    }
    let mut out = vec![0u8; row_w * rows];
    for r in 0..rows {
        out[r * row_w..(r + 1) * row_w]
            .copy_from_slice(&plane.data[r * stride..r * stride + row_w]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jpeg_encode::encode_frame_yuv420p;

    fn video_params(w: u32, h: u32) -> CodecParameters {
        let mut p = CodecParameters::video(CodecId::new("amv_video"));
        p.width = Some(w);
        p.height = Some(h);
        p.pixel_format = Some(PixelFormat::Yuv420P);
        p
    }

    #[test]
    fn factories_reject_missing_geometry() {
        let mut p = CodecParameters::video(CodecId::new("amv_video"));
        assert!(make_decoder(&p).is_err());
        assert!(make_encoder(&p).is_err());
        p.width = Some(16);
        assert!(make_decoder(&p).is_err()); // height still missing
    }

    #[test]
    fn encoder_output_params_declare_yuv420p_geometry() {
        let enc = make_encoder(&video_params(128, 96)).expect("encoder builds");
        let out = enc.output_params();
        assert_eq!(out.width, Some(128));
        assert_eq!(out.height, Some(96));
        assert_eq!(out.pixel_format, Some(PixelFormat::Yuv420P));
        assert_eq!(out.codec_id.as_str(), "amv_video");
    }

    /// Build a synthetic structured YUV420P frame (per-MCU luma gradient).
    fn synth_yuv(w: u32, h: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let cw = (w.div_ceil(2)) as usize;
        let ch = (h.div_ceil(2)) as usize;
        let mut y = vec![0u8; (w * h) as usize];
        for yy in 0..h as usize {
            for xx in 0..w as usize {
                y[yy * w as usize + xx] = (((xx / 16 + yy / 16) * 30 + 60) % 256) as u8;
            }
        }
        (y, vec![128u8; cw * ch], vec![140u8; cw * ch])
    }

    #[test]
    fn decoder_decodes_a_00dc_payload_to_yuv420p_frame() {
        let (w, h) = (33u32, 17u32);
        let (y, cb, cr) = synth_yuv(w, h);
        let payload = encode_frame_yuv420p(w, h, &y, &cb, &cr).expect("encode");

        let mut dec = make_decoder(&video_params(w, h)).expect("decoder builds");
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 12), payload))
            .unwrap();
        let Frame::Video(frame) = dec.receive_frame().unwrap() else {
            panic!("expected a video frame");
        };
        assert_eq!(frame.planes.len(), 3);
        assert_eq!(frame.planes[0].stride, w as usize);
        assert_eq!(frame.planes[0].data.len(), (w * h) as usize);
        assert_eq!(frame.planes[1].stride, w.div_ceil(2) as usize);
        assert_eq!(
            frame.planes[1].data.len(),
            (w.div_ceil(2) * h.div_ceil(2)) as usize
        );
    }

    #[test]
    fn encode_decode_encode_is_a_byte_stable_fixed_point() {
        let (w, h) = (33u32, 17u32);
        let (y, cb, cr) = synth_yuv(w, h);
        let payload = encode_frame_yuv420p(w, h, &y, &cb, &cr).expect("encode");

        // Decode through the trait surface, then re-encode the frame.
        let mut dec = make_decoder(&video_params(w, h)).expect("decoder");
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 12), payload.clone()))
            .unwrap();
        let frame = dec.receive_frame().unwrap();

        let mut enc = make_encoder(&video_params(w, h)).expect("encoder");
        enc.send_frame(&frame).unwrap();
        let pkt = enc.receive_packet().unwrap();
        assert_eq!(
            pkt.data, payload,
            "encode∘decode∘encode must be a byte-stable fixed point"
        );
        assert!(pkt.flags.keyframe, "intra-only: every packet is a keyframe");
    }

    #[test]
    fn encoder_rejects_wrong_plane_count() {
        let mut enc = make_encoder(&video_params(16, 16)).expect("encoder");
        let bad = Frame::Video(VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: 16,
                data: vec![0u8; 256],
            }],
        });
        assert!(enc.send_frame(&bad).is_err());
    }

    #[test]
    fn encoder_handles_padded_strides() {
        // A frame whose plane stride is wider than the row width (trailing
        // pad bytes) must encode the same payload as the tight frame.
        let (w, h) = (17u32, 17u32);
        let (y, cb, cr) = synth_yuv(w, h);
        let tight = encode_frame_yuv420p(w, h, &y, &cb, &cr).expect("encode tight");

        let cw = w.div_ceil(2) as usize;
        let ch = h.div_ceil(2) as usize;
        let pad = 5usize;
        let restride = |src: &[u8], rw: usize, rows: usize| {
            let mut out = vec![0u8; (rw + pad) * rows];
            for r in 0..rows {
                out[r * (rw + pad)..r * (rw + pad) + rw].copy_from_slice(&src[r * rw..r * rw + rw]);
            }
            out
        };
        let frame = Frame::Video(VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: w as usize + pad,
                    data: restride(&y, w as usize, h as usize),
                },
                VideoPlane {
                    stride: cw + pad,
                    data: restride(&cb, cw, ch),
                },
                VideoPlane {
                    stride: cw + pad,
                    data: restride(&cr, cw, ch),
                },
            ],
        });
        let mut enc = make_encoder(&video_params(w, h)).expect("encoder");
        enc.send_frame(&frame).unwrap();
        let pkt = enc.receive_packet().unwrap();
        assert_eq!(
            pkt.data, tight,
            "padded-stride encode must equal the tight-stride encode"
        );
    }

    #[test]
    fn encoder_packet_time_base_follows_frame_rate() {
        use oxideav_core::Rational;
        let (w, h) = (16u32, 16u32);
        let (y, cb, cr) = synth_yuv(w, h);
        let frame = Frame::Video(VideoFrame {
            pts: Some(7),
            planes: vec![
                VideoPlane {
                    stride: w as usize,
                    data: y,
                },
                VideoPlane {
                    stride: (w.div_ceil(2)) as usize,
                    data: cb,
                },
                VideoPlane {
                    stride: (w.div_ceil(2)) as usize,
                    data: cr,
                },
            ],
        });

        // With a 12 fps frame rate the packet base is 1/12.
        let mut params = video_params(w, h);
        params.frame_rate = Some(Rational::new(12, 1));
        let mut enc = make_encoder(&params).expect("encoder");
        enc.send_frame(&frame).unwrap();
        let pkt = enc.receive_packet().unwrap();
        assert_eq!(pkt.time_base, TimeBase::new(1, 12));
        assert_eq!(pkt.pts, Some(7));
        assert_eq!(pkt.dts, Some(7));

        // With no frame rate the base falls back to 1/1.
        let mut enc2 = make_encoder(&video_params(w, h)).expect("encoder");
        enc2.send_frame(&frame).unwrap();
        assert_eq!(
            enc2.receive_packet().unwrap().time_base,
            TimeBase::new(1, 1)
        );
    }

    #[test]
    fn decoder_eof_and_needmore_semantics() {
        let mut dec = make_decoder(&video_params(16, 16)).expect("decoder");
        assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
        dec.flush().unwrap();
        assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    }
}
