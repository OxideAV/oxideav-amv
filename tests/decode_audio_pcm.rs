//! End-to-end milestone validation for the audio side: the AMV `01wb`
//! IMA-ADPCM blocks of a real `comedian.amv`, decoded with
//! [`oxideav_amv::decode_audio_payload`] from the staged §4b step table,
//! produce 16-bit PCM that a **black-box audio tool** (`ffprobe` /
//! `ffmpeg`) reads back as the duration and format the §2/§3b container
//! declares.
//!
//! The crate already proves the decode-sanity numbers internally
//! (2 050 650 mono samples = 93.0 s, sub-0.1 % clip rate — trace §4b).
//! This test closes the loop the way the video `decode_to_pixels` test
//! does: it wraps the decoded PCM in a standard 16-bit mono WAV and hands
//! the bytes to an opaque audio probe, confirming an independent decoder
//! agrees on sample rate, channel count and duration. No audio-tool
//! *source* is read — `ffprobe` is a black-box validator. Skipped when no
//! probe binary is on `PATH`.

use std::path::{Path, PathBuf};
use std::process::Command;

use oxideav_amv::{
    decode_audio_payload, AmvDemuxer, MoviPayload, MoviPayloadIter, AMV_END_TRAILER,
};
use std::fs::File;
use std::io::BufReader;

fn comedian_fixture() -> Option<PathBuf> {
    let crate_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/comedian.amv");
    if crate_path.exists() {
        return Some(crate_path);
    }
    let workspace_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/container/amv/fixtures/comedian.amv");
    if workspace_path.exists() {
        return Some(workspace_path);
    }
    None
}

/// Decode every `01wb` block of the fixture into one contiguous mono PCM
/// buffer, returning `(samples, sample_rate)`.
fn decode_full_audio(path: &Path) -> Vec<i16> {
    let bytes = std::fs::read(path).expect("read comedian fixture");
    let movi_pos = bytes
        .windows(4)
        .position(|w| w == b"movi")
        .expect("movi FOURCC present");
    let trailer_start = bytes.len() - AMV_END_TRAILER.len();
    let movi_body = &bytes[movi_pos + 4..trailer_start];

    let mut pcm = Vec::new();
    for payload in MoviPayloadIter::new(movi_body).filter_map(|r| r.ok()) {
        if let MoviPayload::Audio { body, .. } = payload {
            pcm.extend(decode_audio_payload(body).expect("01wb payload decodes"));
        }
    }
    pcm
}

/// Serialise mono 16-bit PCM to a canonical 44-byte-header WAV.
fn write_wav_mono16(path: &Path, pcm: &[i16], sample_rate: u32) {
    let data_len = (pcm.len() * 2) as u32;
    let byte_rate = sample_rate * 2; // mono, 2 bytes/sample.
    let mut out = Vec::with_capacity(44 + pcm.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, &out).expect("write wav");
}

fn ffprobe_present() -> bool {
    Command::new("ffprobe").arg("-version").output().is_ok()
}

/// Run `ffprobe -show_entries` and return the requested stream entry value.
fn ffprobe_entry(path: &Path, entry: &str) -> Option<String> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            entry,
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path.to_str().unwrap(),
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[test]
fn comedian_audio_decodes_to_pcm_validated_by_ffprobe() {
    let Some(path) = comedian_fixture() else {
        eprintln!("skipping audio-pcm: comedian.amv not staged");
        return;
    };

    // Decode is fixture-truth regardless of validator availability.
    let pcm = decode_full_audio(&path);
    let sample_rate = 22_050u32;
    // §4b internal pin: 2 050 650 mono samples = exactly 93 s.
    assert_eq!(pcm.len(), 2_050_650, "§4b total decoded sample count");
    assert_eq!(
        pcm.len() as u32 / sample_rate,
        93,
        "exactly 93 s @ 22 050 Hz"
    );

    if !ffprobe_present() {
        eprintln!("skipping ffprobe cross-check: ffprobe not on PATH");
        return;
    }

    let dir = std::env::temp_dir().join("oxideav_amv_decode_audio_pcm");
    std::fs::create_dir_all(&dir).unwrap();
    let wav = dir.join("comedian_audio.wav");
    write_wav_mono16(&wav, &pcm, sample_rate);

    // Black-box cross-check: an independent decoder must agree on the
    // declared format and the decoded duration.
    let rate = ffprobe_entry(&wav, "stream=sample_rate").expect("ffprobe sample_rate");
    assert_eq!(rate, "22050", "ffprobe reads back the §3b sample rate");

    let channels = ffprobe_entry(&wav, "stream=channels").expect("ffprobe channels");
    assert_eq!(channels, "1", "ffprobe reads back mono");

    let duration = ffprobe_entry(&wav, "stream=duration")
        .or_else(|| ffprobe_entry(&wav, "format=duration"))
        .expect("ffprobe duration");
    let secs: f64 = duration.parse().expect("numeric duration");
    eprintln!("ffprobe: rate={rate} channels={channels} duration={secs:.3}s");
    // 2 050 650 / 22 050 = 93.0 s exactly.
    assert!(
        (secs - 93.0).abs() < 0.05,
        "ffprobe duration {secs:.3}s should match the §2 container 1:33"
    );
}

/// Demux→PCM via the one-call convenience `AmvDemuxer::decode_audio_packet`
/// — the audio mirror of the video side's
/// `demuxer_decode_video_packet_yields_pixels`. Drives every audio
/// `Packet` the demuxer emits through the convenience and confirms the
/// concatenated PCM is byte-identical to the free-function
/// `decode_audio_payload` path, and that a non-audio packet is rejected.
#[test]
fn demuxer_decode_audio_packet_yields_pcm() {
    use oxideav_core::Demuxer;

    let Some(path) = comedian_fixture() else {
        eprintln!("skipping demuxer decode_audio_packet: comedian.amv not staged");
        return;
    };

    // Ground truth: the free-function whole-track decode.
    let want = decode_full_audio(&path);
    assert_eq!(want.len(), 2_050_650, "§4b total decoded sample count");

    // Drive the demuxer, decoding each audio packet through the
    // convenience and concatenating the PCM.
    let f = File::open(&path).expect("open comedian fixture");
    let mut demuxer = AmvDemuxer::open(BufReader::new(f)).expect("open AMV demuxer");
    assert_eq!(demuxer.audio_format().samples_per_sec, 22_050);

    let mut got: Vec<i16> = Vec::with_capacity(want.len());
    let mut first_video: Option<oxideav_core::Packet> = None;
    let mut audio_blocks = 0u32;
    loop {
        let pkt = match demuxer.next_packet() {
            Ok(p) => p,
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("unexpected demux error: {e}"),
        };
        if pkt.stream_index == 1 {
            got.extend(
                demuxer
                    .decode_audio_packet(&pkt)
                    .expect("audio packet decodes"),
            );
            audio_blocks += 1;
        } else if first_video.is_none() {
            first_video = Some(pkt);
        }
    }

    assert_eq!(audio_blocks, 1116, "§4 1116 audio blocks");
    assert_eq!(
        got, want,
        "demux→PCM convenience matches the free-function decode byte-for-byte"
    );

    // A non-audio (video) packet is rejected by decode_audio_packet.
    let video = first_video.expect("at least one video packet");
    assert!(
        demuxer.decode_audio_packet(&video).is_err(),
        "decode_audio_packet must reject a video packet"
    );
}
