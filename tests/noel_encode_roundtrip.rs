//! Encoder round-trip on the **second device profile**: a real
//! `noel-son-lumiere.amv` (96 × 64 @ 16 fps) decoded, re-encoded with the
//! in-crate §4a/§4b encoders, re-muxed, re-demuxed and re-decoded through
//! the crate's public surface only — the cross-profile companion of the
//! comedian round-trip in `tests/encode_roundtrip.rs`.
//!
//! Profile-specific points this suite pins beyond the comedian loop:
//!
//! * the muxer's §2 duration patch writes the **exact** frame-count
//!   derivation (2928 ÷ 16 = 183 s = 3:03 → packed `0x0303`) — it does
//!   not reproduce the device's one-second truncation (`0x0302`), so the
//!   re-muxed file grades `Exact` where the original grades
//!   `TruncatedByOneSecond` (both device-conformant);
//! * the re-encoded audio headers carry the crate's canonical `+0x02 = 0`
//!   / `+0x03 = 0` bytes (not noel's device-constant `0xAA`), and the §4b
//!   decode is invariant to that difference — the re-decode still matches
//!   because no valid decode honours either byte;
//! * the audio byte-level fixed point holds at 16 fps block sizing
//!   (22 050 ÷ 16 → 1378/1379-sample blocks).
//!
//! Skipped when the fixture is not staged.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use oxideav_amv::{
    decode_audio_payload, decode_frame_from_payload, encode_audio_payload, encode_frame_rgb,
    AmvAudioPreamble, AmvDemuxer, AmvHeader, AmvMuxer, DurationConsistency, MoviPayload,
    MoviPayloadIter, AMVH_BODY_LEN, AMV_END_TRAILER,
};
use oxideav_core::{
    CodecId, CodecParameters, Demuxer, MediaType, Muxer, Packet, PixelFormat, Rational, StreamInfo,
    TimeBase, WriteSeek,
};

fn noel_fixture() -> Option<PathBuf> {
    let crate_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/noel-son-lumiere.amv");
    if crate_path.exists() {
        return Some(crate_path);
    }
    let workspace_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/container/amv/fixtures/noel-son-lumiere.amv");
    if workspace_path.exists() {
        return Some(workspace_path);
    }
    None
}

fn streams_for(header: &AmvHeader, samples_per_sec: u32) -> Vec<StreamInfo> {
    let mut video_params = CodecParameters::video(CodecId::new("mjpeg"));
    video_params.media_type = MediaType::Video;
    video_params.width = Some(header.width);
    video_params.height = Some(header.height);
    video_params.pixel_format = Some(PixelFormat::Yuv420P);
    video_params.frame_rate = Some(Rational::new(header.fps as i64, 1));
    let video = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, header.fps as i64),
        duration: None,
        start_time: Some(0),
        params: video_params,
    };
    let mut audio_params = CodecParameters::audio(CodecId::new("adpcm_amv"));
    audio_params.media_type = MediaType::Audio;
    audio_params.sample_rate = Some(samples_per_sec);
    audio_params.channels = Some(1);
    let audio = StreamInfo {
        index: 1,
        time_base: TimeBase::new(1, samples_per_sec as i64),
        duration: None,
        start_time: Some(0),
        params: audio_params,
    };
    vec![video, audio]
}

fn mux_amv(
    streams: &[StreamInfo],
    video_payloads: &[Vec<u8>],
    audio_payloads: &[Vec<u8>],
    fps: u32,
    sample_rate: u32,
) -> Vec<u8> {
    #[derive(Clone)]
    struct SharedCursor(std::sync::Arc<std::sync::Mutex<Cursor<Vec<u8>>>>);
    impl std::io::Write for SharedCursor {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().write(buf)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.0.lock().unwrap().flush()
        }
    }
    impl std::io::Seek for SharedCursor {
        fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
            self.0.lock().unwrap().seek(pos)
        }
    }

    let shared = SharedCursor(std::sync::Arc::new(std::sync::Mutex::new(Cursor::new(
        Vec::<u8>::new(),
    ))));
    let writer: Box<dyn WriteSeek> = Box::new(shared.clone());
    let mut mux = AmvMuxer::open(writer, streams).expect("open muxer");
    mux.write_header().expect("write header");
    for i in 0..video_payloads.len().max(audio_payloads.len()) {
        if i < video_payloads.len() {
            mux.write_packet(&Packet::new(
                0,
                TimeBase::new(1, fps as i64),
                video_payloads[i].clone(),
            ))
            .expect("write video packet");
        }
        if i < audio_payloads.len() {
            mux.write_packet(&Packet::new(
                1,
                TimeBase::new(1, sample_rate as i64),
                audio_payloads[i].clone(),
            ))
            .expect("write audio packet");
        }
    }
    mux.write_trailer().expect("write trailer");
    drop(mux);
    let bytes = shared.0.lock().unwrap().get_ref().clone();
    bytes
}

#[test]
fn noel_decode_encode_mux_demux_decode_round_trip() {
    let Some(path) = noel_fixture() else {
        eprintln!("skipping noel encode round-trip: noel-son-lumiere.amv not staged");
        return;
    };

    // 1) Decode the real device file to media.
    let bytes = std::fs::read(&path).expect("read noel fixture");
    let header =
        AmvHeader::parse(&bytes[0x20..0x20 + AMVH_BODY_LEN as usize]).expect("amvh parses");
    assert_eq!((header.width, header.height), (96, 64));
    assert_eq!(header.fps, 16);
    let movi_pos = bytes.windows(4).position(|w| w == b"movi").unwrap();
    let trailer_start = bytes.len() - AMV_END_TRAILER.len();
    let movi_body = &bytes[movi_pos + 4..trailer_start];

    let mut frames_rgb: Vec<Vec<u8>> = Vec::new();
    let mut blocks_pcm: Vec<Vec<i16>> = Vec::new();
    for payload in MoviPayloadIter::new(movi_body).filter_map(|r| r.ok()) {
        match payload {
            MoviPayload::Video { body, .. } => {
                frames_rgb.push(
                    decode_frame_from_payload(&header, body)
                        .expect("decode")
                        .rgb,
                );
            }
            MoviPayload::Audio { body, .. } => {
                blocks_pcm.push(decode_audio_payload(body).expect("decode audio"));
            }
            MoviPayload::Other { .. } => {}
        }
    }
    assert_eq!(frames_rgb.len(), 2928);
    assert_eq!(blocks_pcm.len(), 2928);
    // §4b at 16 fps: 22 050 ÷ 16 = 1378.125 → 1378/1379-sample blocks.
    for (i, b) in blocks_pcm.iter().enumerate() {
        assert!(
            (1378..=1379).contains(&b.len()) || i + 1 == blocks_pcm.len(),
            "block {i}: one frame-interval of audio (got {})",
            b.len()
        );
    }

    // 2) Re-encode with the in-crate §4a/§4b encoders.
    let video_payloads: Vec<Vec<u8>> = frames_rgb
        .iter()
        .map(|rgb| encode_frame_rgb(header.width, header.height, rgb).expect("encode frame"))
        .collect();
    let audio_payloads: Vec<Vec<u8>> = blocks_pcm
        .iter()
        .map(|pcm| encode_audio_payload(pcm))
        .collect();

    // Re-encoded audio headers: canonical +0x02 = 0 / +0x03 = 0 (the
    // crate's write side does not reproduce noel's 0xAA device byte —
    // §4b: the byte identifies the producing encoder and no decoder
    // reads it), while the sample count matches the decoded block.
    for (i, (ap, pcm)) in audio_payloads.iter().zip(&blocks_pcm).enumerate() {
        let pre = AmvAudioPreamble::parse(ap).expect("re-encoded preamble");
        assert_eq!(pre.initial_step_index(), 0, "block {i}: canonical +0x02");
        assert_eq!(pre.device_constant_byte(), 0, "block {i}: canonical +0x03");
        assert_eq!(pre.decoded_sample_count as usize, pcm.len(), "block {i}");
    }

    // 3) Mux; the §2 duration patch is the EXACT derivation, not the
    //    device's truncation.
    let streams = streams_for(&header, 22_050);
    let amv = mux_amv(
        &streams,
        &video_payloads,
        &audio_payloads,
        header.fps,
        22_050,
    );
    assert_eq!(&amv[0..4], b"RIFF");
    assert_eq!(&amv[8..12], b"AMV ");
    assert_eq!(&amv[amv.len() - 8..], &AMV_END_TRAILER);
    let patched_dur = u32::from_le_bytes(amv[0x54..0x58].try_into().unwrap());
    assert_eq!(
        patched_dur, 0x0000_0303,
        "muxer writes the exact 2928 ÷ 16 = 3:03, not the device's truncated 3:02"
    );

    // 4) Re-demux + re-decode.
    let mut d = AmvDemuxer::open_strict(Cursor::new(amv)).expect("strict re-open muxed AMV");
    assert_eq!((d.header().width, d.header().height), (96, 64));
    assert_eq!(d.header().fps, 16);
    let rt_header = *d.header();
    let mut redec_frames: Vec<Vec<u8>> = Vec::new();
    let mut redec_blocks: Vec<Vec<i16>> = Vec::new();
    loop {
        match d.next_packet() {
            Ok(p) if p.stream_index == 0 => {
                redec_frames.push(
                    decode_frame_from_payload(&rt_header, &p.data)
                        .expect("redecode")
                        .rgb,
                );
            }
            Ok(p) if p.stream_index == 1 => {
                redec_blocks.push(decode_audio_payload(&p.data).expect("redecode audio"));
            }
            Ok(_) => panic!("unexpected stream index"),
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("re-demux walk error: {e:?}"),
        }
    }
    assert_eq!(redec_frames.len(), 2928);
    assert_eq!(redec_blocks.len(), 2928);
    assert!(d.movi_interleave_balanced());
    // The re-muxed header agrees with its own walk EXACTLY (grade
    // `Exact`), where the original device file grades truncated.
    assert!(d.duration_consistent_with_drained_frames());
    assert_eq!(
        d.duration_consistency_with_drained_frames(),
        DurationConsistency::Exact
    );

    // 5) Stability: the re-decode tracks the source decode globally.
    let mut video_sum_abs = 0f64;
    let mut video_count = 0u64;
    for (a, b) in frames_rgb.iter().zip(&redec_frames) {
        assert_eq!(a.len(), b.len());
        for (&pa, &pb) in a.iter().zip(b.iter()) {
            video_sum_abs += pa.abs_diff(pb) as f64;
            video_count += 1;
        }
    }
    let video_mae = video_sum_abs / video_count as f64;
    assert!(
        video_mae < 3.0,
        "video round-trip MAE {video_mae}/channel too high on the noel profile"
    );

    let mut audio_sum_abs = 0f64;
    let mut audio_count = 0u64;
    for (a, b) in blocks_pcm.iter().zip(&redec_blocks) {
        assert_eq!(a.len(), b.len(), "block length preserved");
        for (&sa, &sb) in a.iter().zip(b.iter()) {
            audio_sum_abs += (sa as f64 - sb as f64).abs();
            audio_count += 1;
        }
    }
    let audio_mae = audio_sum_abs / audio_count as f64;
    assert!(
        audio_mae < 200.0,
        "audio round-trip MAE {audio_mae} too high on the noel profile"
    );

    // 6) Audio byte-level fixed point at 16 fps block sizing: re-encoding
    //    the re-decoded PCM reproduces the exact `01wb` bytes we muxed.
    for (i, (ap, pcm)) in audio_payloads.iter().zip(&redec_blocks).enumerate() {
        let re = encode_audio_payload(pcm);
        assert_eq!(
            ap, &re,
            "audio payload {i}: encode∘decode is not byte-idempotent"
        );
    }
}
