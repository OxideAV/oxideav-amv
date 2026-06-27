//! End-to-end registry-driven codec validation: the `adpcm_amv` and
//! `amv_video` codecs that [`oxideav_amv::register`] installs decode the
//! real device-origin `comedian.amv` bytes through `oxideav-core`'s
//! `Decoder` trait contract — not just the direct free functions.
//!
//! This is the integration-level counterpart of the in-crate unit tests
//! in `codec_audio` / `codec_video`: it opens the staged fixture with
//! the real [`AmvDemuxer`], pulls packets, resolves the registered
//! decoder factories out of a `RuntimeContext`, and confirms the trait
//! path produces the same frames the direct
//! [`decode_audio_payload`](oxideav_amv::decode_audio_payload) /
//! [`decode_frame_yuv420p_from_payload`](oxideav_amv::decode_frame_yuv420p_from_payload)
//! calls produce on the same bytes.
//!
//! Skips cleanly when the fixture is not staged.

use std::path::{Path, PathBuf};

use oxideav_amv::{
    decode_audio_payload, decode_frame_yuv420p, register, AmvHeader, AmvVideoFrame, MoviPayload,
    MoviPayloadIter, AMVH_BODY_LEN, AMV_END_TRAILER, AUDIO_CODEC_ID, VIDEO_DIRECT_CODEC_ID,
};
use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, RuntimeContext, SampleFormat, TimeBase,
};

fn comedian_fixture() -> Option<PathBuf> {
    let crate_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/comedian.amv");
    if crate_path.exists() {
        return Some(crate_path);
    }
    let workspace_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/container/amv/fixtures/comedian.amv");
    workspace_path.exists().then_some(workspace_path)
}

/// Parse the §2 header and the raw `00dc` / `01wb` payloads of the
/// fixture's `movi` body.
fn fixture_payloads(path: &Path) -> (AmvHeader, Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let bytes = std::fs::read(path).expect("read comedian fixture");
    let header =
        AmvHeader::parse(&bytes[0x20..0x20 + AMVH_BODY_LEN as usize]).expect("amvh parses");
    let movi_pos = bytes
        .windows(4)
        .position(|w| w == b"movi")
        .expect("movi present");
    let trailer_start = bytes.len() - AMV_END_TRAILER.len();
    let movi_body = &bytes[movi_pos + 4..trailer_start];

    let mut video = Vec::new();
    let mut audio = Vec::new();
    for payload in MoviPayloadIter::new(movi_body).filter_map(|r| r.ok()) {
        match payload {
            MoviPayload::Video { body, .. } => video.push(body.to_vec()),
            MoviPayload::Audio { body, .. } => audio.push(body.to_vec()),
            MoviPayload::Other { .. } => {}
        }
    }
    (header, video, audio)
}

fn audio_params() -> CodecParameters {
    let mut p = CodecParameters::audio(CodecId::new(AUDIO_CODEC_ID));
    p.channels = Some(1);
    p.sample_rate = Some(22_050);
    p.sample_format = Some(SampleFormat::S16);
    p
}

fn video_params(header: &AmvHeader) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new(VIDEO_DIRECT_CODEC_ID));
    p.width = Some(header.width);
    p.height = Some(header.height);
    p.pixel_format = Some(PixelFormat::Yuv420P);
    p
}

#[test]
fn registry_video_decoder_matches_direct_decode_on_real_frames() {
    let Some(path) = comedian_fixture() else {
        eprintln!("skipping registry video path: comedian.amv not staged");
        return;
    };
    let (header, video_payloads, _) = fixture_payloads(&path);
    assert!(!video_payloads.is_empty(), "fixture has video frames");

    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let params = video_params(&header);

    // Sweep a spread of frames through the registry decoder and compare
    // to the direct YUV decode of the same bytes.
    let n = video_payloads.len();
    for &i in &[0usize, 1, 5, 50, n / 2, n - 1] {
        let payload = &video_payloads[i];
        let mut dec = ctx
            .codecs
            .first_decoder(&params)
            .expect("registry resolves amv_video decoder");
        let pkt = Packet::new(0, TimeBase::new(1, header.fps as i64), payload.clone());
        dec.send_packet(&pkt).expect("send video packet");
        let Frame::Video(frame) = dec.receive_frame().expect("receive video frame") else {
            panic!("expected a video frame for frame {i}");
        };

        // Direct reference decode.
        let bound = AmvVideoFrame::bind_strict(&header, payload).expect("bind frame");
        let yuv = decode_frame_yuv420p(&bound).expect("direct yuv decode");

        assert_eq!(frame.planes.len(), 3, "frame {i}: YUV420P has 3 planes");
        assert_eq!(frame.planes[0].stride, header.width as usize);
        assert_eq!(frame.planes[0].data, yuv.y, "frame {i}: Y plane mismatch");
        assert_eq!(frame.planes[1].data, yuv.cb, "frame {i}: Cb plane mismatch");
        assert_eq!(frame.planes[2].data, yuv.cr, "frame {i}: Cr plane mismatch");
    }
}

#[test]
fn registry_audio_decoder_matches_direct_decode_on_real_blocks() {
    let Some(path) = comedian_fixture() else {
        eprintln!("skipping registry audio path: comedian.amv not staged");
        return;
    };
    let (_, _, audio_payloads) = fixture_payloads(&path);
    assert!(!audio_payloads.is_empty(), "fixture has audio blocks");

    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    let params = audio_params();

    let n = audio_payloads.len();
    for &i in &[0usize, 1, 50, n / 2, n - 1] {
        let payload = &audio_payloads[i];
        let mut dec = ctx
            .codecs
            .first_decoder(&params)
            .expect("registry resolves adpcm_amv decoder");
        let pkt = Packet::new(1, TimeBase::new(1, 22_050), payload.clone());
        dec.send_packet(&pkt).expect("send audio packet");
        let Frame::Audio(frame) = dec.receive_frame().expect("receive audio frame") else {
            panic!("expected an audio frame for block {i}");
        };

        let direct = decode_audio_payload(payload).expect("direct audio decode");
        assert_eq!(
            frame.samples as usize,
            direct.len(),
            "block {i}: sample count"
        );
        assert_eq!(frame.data.len(), 1, "block {i}: mono → one plane");
        let got: Vec<i16> = frame.data[0]
            .chunks_exact(2)
            .map(|p| i16::from_le_bytes([p[0], p[1]]))
            .collect();
        assert_eq!(got, direct, "block {i}: PCM mismatch vs direct decode");
    }
}

#[test]
fn registry_full_stream_decodes_through_demuxer_and_codecs() {
    use oxideav_amv::AmvDemuxer;
    use oxideav_core::Demuxer;

    let Some(path) = comedian_fixture() else {
        eprintln!("skipping full-stream registry path: comedian.amv not staged");
        return;
    };

    let f = std::fs::File::open(&path).expect("open fixture");
    let mut demuxer = AmvDemuxer::open(std::io::BufReader::new(f)).expect("open demuxer");
    let header = *demuxer.header();

    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    // Zero-config path: the demuxer's own StreamInfo params carry the
    // direct `amv_video` / `adpcm_amv` codec ids, so a registered
    // RuntimeContext resolves a working decoder for each stream with no
    // hand-built CodecParameters.
    let vparams = demuxer.streams()[0].params.clone();
    let aparams = demuxer.streams()[1].params.clone();
    assert_eq!(vparams.codec_id.as_str(), VIDEO_DIRECT_CODEC_ID);
    assert_eq!(aparams.codec_id.as_str(), AUDIO_CODEC_ID);

    let mut video_frames = 0usize;
    let mut audio_frames = 0usize;
    let mut total_audio_samples = 0u64;

    // Drive the first slice of the interleaved stream end-to-end: demux
    // packet → registry decoder for its stream → frame. Bound the walk so
    // the test stays fast but still crosses many video/audio pairs.
    for _ in 0..200 {
        let pkt = match demuxer.next_packet() {
            Ok(p) => p,
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("demux error: {e:?}"),
        };
        match pkt.stream_index {
            0 => {
                let mut dec = ctx.codecs.first_decoder(&vparams).expect("video decoder");
                dec.send_packet(&pkt).expect("send video");
                let Frame::Video(frame) = dec.receive_frame().expect("video frame") else {
                    panic!("expected video frame");
                };
                assert_eq!(frame.planes.len(), 3);
                assert_eq!(
                    frame.planes[0].data.len(),
                    (header.width * header.height) as usize
                );
                video_frames += 1;
            }
            1 => {
                let mut dec = ctx.codecs.first_decoder(&aparams).expect("audio decoder");
                dec.send_packet(&pkt).expect("send audio");
                let Frame::Audio(frame) = dec.receive_frame().expect("audio frame") else {
                    panic!("expected audio frame");
                };
                total_audio_samples += frame.samples as u64;
                audio_frames += 1;
            }
            other => panic!("unexpected stream index {other}"),
        }
    }

    assert!(video_frames > 0, "decoded at least one video frame");
    assert!(audio_frames > 0, "decoded at least one audio block");
    assert!(
        total_audio_samples > 0,
        "decoded a non-empty PCM stream through the registry path"
    );
}
