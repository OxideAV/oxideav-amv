//! `oxideav-core` [`Decoder`] / [`Encoder`] trait surface for the AMV
//! intrinsic audio codec (`adpcm_amv`).
//!
//! The byte-level decode / encode of an `01wb` payload already lives in
//! [`crate::adpcm`] / [`crate::adpcm_encode`] as free functions
//! ([`decode_audio_payload`](crate::decode_audio_payload) /
//! [`encode_audio_payload`](crate::encode_audio_payload)); this module is
//! the thin registry-facing wrapper that lets those run through
//! `oxideav-core`'s packet→frame / frame→packet contract, so the AMV
//! audio stream the demuxer emits can be driven by the pipeline /
//! `make_decoder` path rather than only by direct function calls.
//!
//! # The `adpcm_amv` profile (trace §3b / §4b)
//!
//! * **Mono, 22 050 Hz, 16-bit signed** decoded PCM — the §3b
//!   `WAVEFORMATEX` rate (the on-disk `wFormatTag = 1` "PCM" claim is a
//!   placeholder; the real payload is 4-bit IMA-ADPCM nibbles, §4b).
//! * One `01wb` payload per packet: an 8-byte §4b preamble (`int16`
//!   predictor seed, ignored step index, `u32` decoded sample count)
//!   followed by low-nibble-first packed 4-bit IMA-ADPCM.
//! * Each block is **self-contained** — the predictor re-seeds from the
//!   preamble and the step index resets to 0 — so the decoder carries no
//!   cross-packet state and `reset` is a no-op beyond draining.
//!
//! The decoder emits one interleaved-mono [`AudioFrame`] (`data` has a
//! single plane of little-endian `i16` samples) per packet; the encoder
//! is the byte-inverse, packing one `01wb` payload per input frame.

use std::collections::VecDeque;

use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Decoder, Encoder, Error, Frame, MediaType, Packet,
    Result, SampleFormat, TimeBase,
};

use crate::adpcm::decode_audio_payload;
use crate::adpcm_encode::encode_audio_payload;

/// Canonical decoded sample rate of the AMV audio stream (trace §3b
/// `WAVEFORMATEX nSamplesPerSec`).
pub const AMV_AUDIO_SAMPLE_RATE: u32 = 22_050;

/// AMV audio is always single-channel (trace §3b `nChannels = 1`).
pub const AMV_AUDIO_CHANNELS: u16 = 1;

/// Serialize a slice of mono `i16` samples to interleaved little-endian
/// bytes — the [`AudioFrame`] plane layout for [`SampleFormat::S16`].
fn samples_to_le_bytes(samples: &[i16]) -> Vec<u8> {
    let mut out = vec![0u8; samples.len() * 2];
    for (s, dst) in samples.iter().zip(out.chunks_exact_mut(2)) {
        dst.copy_from_slice(&s.to_le_bytes());
    }
    out
}

/// Parse interleaved little-endian S16 bytes back to mono `i16` samples.
fn le_bytes_to_samples(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|p| i16::from_le_bytes([p[0], p[1]]))
        .collect()
}

// ───────────────────────────── decoder ─────────────────────────────

/// Build a boxed [`Decoder`] for the AMV `adpcm_amv` audio codec.
///
/// Direct-factory entry point: [`crate::register_codecs`] installs this
/// same function into the registry, so callers that want to skip the
/// registry lookup may invoke it directly with manually-built `params`.
/// The codec is mono-only by construction (trace §3b); a non-1 channel
/// count in `params` is rejected so a mis-described stream surfaces
/// early rather than mis-decoding.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    let channels = params.channels.unwrap_or(AMV_AUDIO_CHANNELS);
    if channels != AMV_AUDIO_CHANNELS {
        return Err(Error::unsupported(format!(
            "adpcm_amv decoder: AMV audio is mono only (got {channels} channels)"
        )));
    }
    Ok(Box::new(AmvAudioDecoder {
        codec_id: params.codec_id.clone(),
        pending: None,
        eof: false,
    }))
}

/// AMV `adpcm_amv` audio decoder: one `01wb` payload in, one mono S16
/// [`AudioFrame`] out. Stateless across packets (each §4b block
/// re-seeds), so a single `send_packet` → `receive_frame` cycle fully
/// drains one block.
#[derive(Debug)]
pub struct AmvAudioDecoder {
    codec_id: CodecId,
    pending: Option<Packet>,
    eof: bool,
}

impl Decoder for AmvAudioDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::other(
                "adpcm_amv decoder: call receive_frame before sending another packet",
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
        let samples = decode_audio_payload(&pkt.data).map_err(Error::from)?;
        let data = samples_to_le_bytes(&samples);
        Ok(Frame::Audio(AudioFrame {
            samples: samples.len() as u32,
            pts: pkt.pts,
            data: vec![data],
        }))
    }

    fn flush(&mut self) -> Result<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        // No cross-packet state: drop any buffered input and clear EOF
        // so the next send_packet decodes as if first.
        self.pending = None;
        self.eof = false;
        Ok(())
    }
}

// ───────────────────────────── encoder ─────────────────────────────

/// Build a boxed [`Encoder`] for the AMV `adpcm_amv` audio codec.
///
/// Direct-factory counterpart to [`make_decoder`]. Input must be mono
/// [`SampleFormat::S16`] (the only thing AMV audio represents). The
/// returned [`Encoder::output_params`] declare the §3b stream shape
/// (mono / 22 050 Hz / S16) so a downstream [`crate::AmvMuxer`] writes
/// the correct `WAVEFORMATEX`.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    let channels = params.channels.unwrap_or(AMV_AUDIO_CHANNELS);
    if channels != AMV_AUDIO_CHANNELS {
        return Err(Error::unsupported(format!(
            "adpcm_amv encoder: AMV audio is mono only (got {channels} channels)"
        )));
    }
    let sample_format = params.sample_format.unwrap_or(SampleFormat::S16);
    if sample_format != SampleFormat::S16 {
        return Err(Error::unsupported(format!(
            "adpcm_amv encoder: input sample format {sample_format:?} not supported (need S16)"
        )));
    }
    let sample_rate = params.sample_rate.unwrap_or(AMV_AUDIO_SAMPLE_RATE);

    let mut output = params.clone();
    output.media_type = MediaType::Audio;
    output.codec_id = params.codec_id.clone();
    output.channels = Some(AMV_AUDIO_CHANNELS);
    output.sample_rate = Some(sample_rate);
    output.sample_format = Some(SampleFormat::S16);

    Ok(Box::new(AmvAudioEncoder {
        output,
        time_base: TimeBase::new(1, sample_rate as i64),
        queue: VecDeque::new(),
    }))
}

/// AMV `adpcm_amv` audio encoder: one mono S16 [`AudioFrame`] in, one
/// `01wb` payload (8-byte §4b preamble + nibble body) out.
#[derive(Debug)]
pub struct AmvAudioEncoder {
    output: CodecParameters,
    time_base: TimeBase,
    queue: VecDeque<Packet>,
}

impl Encoder for AmvAudioEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.output.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let Frame::Audio(a) = frame else {
            return Err(Error::invalid("adpcm_amv encoder: audio frames only"));
        };
        let bytes = a
            .data
            .first()
            .ok_or_else(|| Error::invalid("adpcm_amv encoder: empty frame"))?;
        if bytes.len() % 2 != 0 {
            return Err(Error::invalid(
                "adpcm_amv encoder: S16 plane byte count must be even",
            ));
        }
        let samples = le_bytes_to_samples(bytes);
        let payload = encode_audio_payload(&samples);
        let mut pkt = Packet::new(0, self.time_base, payload);
        pkt.pts = a.pts;
        pkt.dts = a.pts;
        pkt.duration = Some(samples.len() as i64);
        pkt.flags.keyframe = true;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adpcm_encode::encode_audio_payload;

    fn audio_params() -> CodecParameters {
        let mut p = CodecParameters::audio(CodecId::new("adpcm_amv"));
        p.channels = Some(1);
        p.sample_rate = Some(AMV_AUDIO_SAMPLE_RATE);
        p.sample_format = Some(SampleFormat::S16);
        p
    }

    #[test]
    fn decoder_rejects_stereo_params() {
        let mut p = audio_params();
        p.channels = Some(2);
        assert!(make_decoder(&p).is_err());
    }

    #[test]
    fn encoder_rejects_non_s16() {
        let mut p = audio_params();
        p.sample_format = Some(SampleFormat::U8);
        assert!(make_encoder(&p).is_err());
    }

    #[test]
    fn encoder_output_params_declare_mono_22050_s16() {
        let enc = make_encoder(&audio_params()).expect("encoder builds");
        let out = enc.output_params();
        assert_eq!(out.channels, Some(1));
        assert_eq!(out.sample_rate, Some(AMV_AUDIO_SAMPLE_RATE));
        assert_eq!(out.sample_format, Some(SampleFormat::S16));
        assert_eq!(out.codec_id.as_str(), "adpcm_amv");
    }

    #[test]
    fn decoder_decodes_an_01wb_payload_to_mono_s16() {
        // Build a real §4b payload from known samples, then decode it
        // through the trait surface and check the frame shape + that the
        // bytes match a direct `decode_audio_payload`.
        let samples: Vec<i16> = (0..256).map(|i| ((i * 37) % 4000 - 2000) as i16).collect();
        let payload = encode_audio_payload(&samples);

        let mut dec = make_decoder(&audio_params()).expect("decoder builds");
        let pkt = Packet::new(
            1,
            TimeBase::new(1, AMV_AUDIO_SAMPLE_RATE as i64),
            payload.clone(),
        );
        dec.send_packet(&pkt).expect("send packet");
        let Frame::Audio(frame) = dec.receive_frame().expect("receive frame") else {
            panic!("expected an audio frame");
        };

        let direct = decode_audio_payload(&payload).expect("direct decode");
        assert_eq!(frame.samples as usize, direct.len());
        assert_eq!(frame.data.len(), 1, "mono → one plane");
        let round = le_bytes_to_samples(&frame.data[0]);
        assert_eq!(round, direct, "trait decode must equal direct decode");
        assert_eq!(frame.samples as usize, samples.len());
    }

    #[test]
    fn decoder_then_encoder_round_trip_is_byte_stable() {
        // encode → decode-frame → re-encode the decoded PCM must reproduce
        // the identical `01wb` payload: the §4b encoder tracks the
        // decoder's reconstructed predictor, so encode∘decode∘encode is a
        // fixed point.
        let samples: Vec<i16> = (0..512)
            .map(|i| (((i * 91) % 6000) - 3000) as i16)
            .collect();
        let payload = encode_audio_payload(&samples);

        let mut dec = make_decoder(&audio_params()).expect("decoder builds");
        dec.send_packet(&Packet::new(
            0,
            TimeBase::new(1, AMV_AUDIO_SAMPLE_RATE as i64),
            payload.clone(),
        ))
        .unwrap();
        let frame = dec.receive_frame().unwrap();

        let mut enc = make_encoder(&audio_params()).expect("encoder builds");
        enc.send_frame(&frame).unwrap();
        let pkt = enc.receive_packet().unwrap();
        assert_eq!(
            pkt.data, payload,
            "encode∘decode∘encode must be a byte-stable fixed point"
        );
    }

    #[test]
    fn decoder_eof_after_flush() {
        let mut dec = make_decoder(&audio_params()).expect("decoder builds");
        dec.flush().unwrap();
        assert!(matches!(dec.receive_frame(), Err(Error::Eof)));
    }

    #[test]
    fn decoder_needmore_before_first_packet() {
        let mut dec = make_decoder(&audio_params()).expect("decoder builds");
        assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
    }

    #[test]
    fn encoder_needmore_when_queue_empty() {
        let mut enc = make_encoder(&audio_params()).expect("encoder builds");
        assert!(matches!(enc.receive_packet(), Err(Error::NeedMore)));
    }
}
