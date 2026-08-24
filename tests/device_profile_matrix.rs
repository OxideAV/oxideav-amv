//! Device-profile matrix: the corpus survey in the trace's fixtures
//! notes records the published 28-file corpus spanning geometries
//! 96 × 64, 128 × 96, 128 × 128 and 160 × 120 at 8, 10, 12 and 16 fps
//! (all 22 050 Hz mono). The two staged fixtures pin two of those
//! profiles on real bytes; this suite closes the rest of the observed
//! parameter space synthetically through the crate's own write → read
//! surface: for every (geometry × fps) cell, encode synthetic video +
//! audio, mux, strict-open, demux, decode, and check the §2/§3b/§4
//! invariants and media fidelity.
//!
//! 160 × 120 is the corpus's only non-multiple-of-16 geometry (7.5 MCU
//! rows): here it exercises the **encoder's** 16×16 edge-replication pad
//! and the decoder's crop at a real device geometry, complementing the
//! decode-side synthetic-geometry unit harness and the trace's
//! real-bytes confirmation that such files decode under the ordinary
//! JPEG padding rule.
//!
//! Fully synthetic — no fixture needed, never skips.

use std::io::Cursor;

use oxideav_amv::{
    decode_audio_payload, decode_frame_from_payload, encode_audio_payload, encode_frame_rgb,
    AmvDemuxer, AmvMuxer, DurationConsistency,
};
use oxideav_core::{
    CodecId, CodecParameters, Demuxer, MediaType, Muxer, Packet, PixelFormat, Rational, StreamInfo,
    TimeBase, WriteSeek,
};

/// Synthetic natural-ish content: a smooth two-axis gradient with a
/// moving diagonal band, distinct per frame index — compresses well at
/// the fixed §4a tables while still exercising chroma + AC coefficients.
fn synth_rgb(width: u32, height: u32, frame_idx: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width.max(1)) as u8;
            let g = ((y * 255) / height.max(1)) as u8;
            let b = (((x + y + 7 * frame_idx) % 255) / 2) as u8;
            rgb.extend_from_slice(&[r, g, b]);
        }
    }
    rgb
}

/// One frame-interval of a synthetic tone at 22 050 Hz for the given
/// fps cell (the §4b per-block sample budget: `samples_per_sec / fps`).
fn synth_pcm(fps: u32, block_idx: u32) -> Vec<i16> {
    let n = 22_050 / fps;
    (0..n)
        .map(|i| {
            let t = (block_idx * n + i) as f64 / 22_050.0;
            // A modest-amplitude two-tone signal, well inside i16.
            ((t * 440.0 * std::f64::consts::TAU).sin() * 6000.0
                + (t * 97.0 * std::f64::consts::TAU).sin() * 2500.0) as i16
        })
        .collect()
}

fn streams_for(width: u32, height: u32, fps: u32) -> Vec<StreamInfo> {
    let mut video_params = CodecParameters::video(CodecId::new("mjpeg"));
    video_params.media_type = MediaType::Video;
    video_params.width = Some(width);
    video_params.height = Some(height);
    video_params.pixel_format = Some(PixelFormat::Yuv420P);
    video_params.frame_rate = Some(Rational::new(fps as i64, 1));
    let video = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, fps as i64),
        duration: None,
        start_time: Some(0),
        params: video_params,
    };
    let mut audio_params = CodecParameters::audio(CodecId::new("adpcm_amv"));
    audio_params.media_type = MediaType::Audio;
    audio_params.sample_rate = Some(22_050);
    audio_params.channels = Some(1);
    let audio = StreamInfo {
        index: 1,
        time_base: TimeBase::new(1, 22_050),
        duration: None,
        start_time: Some(0),
        params: audio_params,
    };
    vec![video, audio]
}

fn mux_pairs(
    width: u32,
    height: u32,
    fps: u32,
    video_payloads: &[Vec<u8>],
    audio_payloads: &[Vec<u8>],
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
    let streams = streams_for(width, height, fps);
    let mut mux = AmvMuxer::open(writer, &streams).expect("open muxer");
    mux.write_header().expect("write header");
    for (v, a) in video_payloads.iter().zip(audio_payloads) {
        mux.write_packet(&Packet::new(0, TimeBase::new(1, fps as i64), v.clone()))
            .expect("write video packet");
        mux.write_packet(&Packet::new(1, TimeBase::new(1, 22_050), a.clone()))
            .expect("write audio packet");
    }
    mux.write_trailer().expect("write trailer");
    drop(mux);
    let bytes = shared.0.lock().unwrap().get_ref().clone();
    bytes
}

/// The full observed parameter space, one cell at a time: encode → mux
/// → strict open → drain → decode, checking geometry, fps wiring
/// (`dwMicroSecPerFrame = 1e6 / fps`), the §4b per-block sample budget,
/// interleave balance, the exact §2 duration grade, and media fidelity.
#[test]
fn corpus_observed_profile_matrix_round_trips() {
    // Corpus survey axes (fixtures notes): four geometries, four rates.
    let geometries = [(96u32, 64u32), (128, 96), (128, 128), (160, 120)];
    let rates = [8u32, 10, 12, 16];
    const PAIRS: u32 = 3;

    for &(width, height) in &geometries {
        for &fps in &rates {
            // --- encode ------------------------------------------------
            let sources: Vec<Vec<u8>> = (0..PAIRS).map(|i| synth_rgb(width, height, i)).collect();
            let video_payloads: Vec<Vec<u8>> = sources
                .iter()
                .map(|rgb| encode_frame_rgb(width, height, rgb).expect("encode frame"))
                .collect();
            let pcm_blocks: Vec<Vec<i16>> = (0..PAIRS).map(|i| synth_pcm(fps, i)).collect();
            let audio_payloads: Vec<Vec<u8>> = pcm_blocks
                .iter()
                .map(|pcm| encode_audio_payload(pcm))
                .collect();

            // --- mux + strict open ------------------------------------
            let amv = mux_pairs(width, height, fps, &video_payloads, &audio_payloads);
            let mut d = AmvDemuxer::open_strict(Cursor::new(amv))
                .unwrap_or_else(|e| panic!("{width}x{height}@{fps}: strict open: {e:?}"));
            assert_eq!(d.header().width, width, "{width}x{height}@{fps}");
            assert_eq!(d.header().height, height, "{width}x{height}@{fps}");
            assert_eq!(d.header().fps, fps, "{width}x{height}@{fps}");
            assert_eq!(
                d.header().micros_per_frame,
                1_000_000 / fps,
                "{width}x{height}@{fps}: §2 dwMicroSecPerFrame"
            );
            assert_eq!(d.audio_format().samples_per_sec, 22_050);
            assert_eq!(
                d.audio_format().frame_interval_samples(fps),
                22_050 / fps,
                "{width}x{height}@{fps}: §4b per-block sample budget"
            );

            // --- drain + decode ---------------------------------------
            let rt_header = *d.header();
            let mut n_video = 0u32;
            let mut n_audio = 0u32;
            let mut video_mae_sum = 0f64;
            let mut video_px = 0u64;
            loop {
                match d.next_packet() {
                    Ok(p) if p.stream_index == 0 => {
                        let f = decode_frame_from_payload(&rt_header, &p.data)
                            .unwrap_or_else(|e| panic!("{width}x{height}@{fps}: decode: {e:?}"));
                        assert_eq!((f.width, f.height), (width, height));
                        let src = &sources[n_video as usize];
                        assert_eq!(f.rgb.len(), src.len());
                        for (&a, &b) in f.rgb.iter().zip(src) {
                            video_mae_sum += a.abs_diff(b) as f64;
                            video_px += 1;
                        }
                        n_video += 1;
                    }
                    Ok(p) if p.stream_index == 1 => {
                        let pcm = decode_audio_payload(&p.data).expect("audio decodes");
                        let src = &pcm_blocks[n_audio as usize];
                        assert_eq!(pcm.len(), src.len(), "block sample count");
                        // IMA-ADPCM at ~6 dB/step tracks a smooth tone
                        // closely; a coarse global bound guards against
                        // any per-profile decode derailment.
                        let mae = pcm
                            .iter()
                            .zip(src)
                            .map(|(&x, &y)| (x as f64 - y as f64).abs())
                            .sum::<f64>()
                            / pcm.len() as f64;
                        assert!(mae < 300.0, "{width}x{height}@{fps}: audio MAE {mae:.1}");
                        // One decode∘encode loop reaches the byte fixed
                        // point in every profile cell.
                        assert_eq!(
                            encode_audio_payload(&pcm),
                            p.data.to_vec(),
                            "{width}x{height}@{fps}: audio byte fixed point"
                        );
                        n_audio += 1;
                    }
                    Ok(_) => panic!("unexpected stream index"),
                    Err(oxideav_core::Error::Eof) => break,
                    Err(e) => panic!("{width}x{height}@{fps}: walk: {e:?}"),
                }
            }
            assert_eq!((n_video, n_audio), (PAIRS, PAIRS));
            assert!(d.movi_interleave_balanced());
            assert_eq!(
                d.duration_consistency_with_drained_frames(),
                DurationConsistency::Exact,
                "{width}x{height}@{fps}: muxer-patched §2 duration is the exact derivation"
            );

            // The smooth-gradient content codes near-transparently at
            // the fixed q≈50 tables; a generous bound still catches a
            // geometry-dependent pad/crop bug (a leaked pad row is tens
            // of levels off across the bottom band).
            let video_mae = video_mae_sum / video_px as f64;
            assert!(
                video_mae < 6.0,
                "{width}x{height}@{fps}: video MAE {video_mae:.2}"
            );
        }
    }
}

/// The non-mod-16 device geometry, pinned harder: at 160 × 120 the
/// bottom MCU row is half-covered (120 = 7.5 × 16), so an encoder pad
/// or decoder crop bug concentrates its error in the last 8 pixel rows.
/// Compare the bottom band's error against the whole-frame error to
/// prove no pad leakage at exactly the geometry the device corpus
/// exercises.
#[test]
fn non_mod16_device_geometry_has_no_bottom_band_pad_leakage() {
    let (width, height) = (160u32, 120u32);
    let src = synth_rgb(width, height, 1);
    let payload = encode_frame_rgb(width, height, &src).expect("encode");
    let header = oxideav_amv::AmvHeader {
        micros_per_frame: 1_000_000 / 12,
        width,
        height,
        fps: 12,
        flag_one: 1,
        reserved_30: 0,
        duration_packed: 0,
    };
    let dec = decode_frame_from_payload(&header, &payload).expect("decode");
    assert_eq!((dec.width, dec.height), (width, height));

    let row_bytes = (width * 3) as usize;
    let band_rows = 8usize; // the half-MCU tail of the 8th MCU row
    let whole_mae = dec
        .rgb
        .iter()
        .zip(&src)
        .map(|(&a, &b)| a.abs_diff(b) as f64)
        .sum::<f64>()
        / dec.rgb.len() as f64;
    let band_start = (height as usize - band_rows) * row_bytes;
    let band_mae = dec.rgb[band_start..]
        .iter()
        .zip(&src[band_start..])
        .map(|(&a, &b)| a.abs_diff(b) as f64)
        .sum::<f64>()
        / (band_rows * row_bytes) as f64;
    assert!(whole_mae < 6.0, "whole-frame MAE {whole_mae:.2}");
    assert!(
        band_mae < whole_mae * 2.0 + 2.0,
        "bottom-band MAE {band_mae:.2} vs whole {whole_mae:.2} — pad leakage at 7.5 MCU rows"
    );
}
