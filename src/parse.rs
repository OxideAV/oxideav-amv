//! Byte-level AMV parser. Pure logic, no I/O — every function takes a
//! borrowed byte slice and returns either a structured view or an
//! [`crate::AmvDemuxerError`].
//!
//! The parser walks exactly the byte layout documented in
//! `docs/container/amv/amv-container-trace.md`:
//!
//! 1. Top-level `RIFF .... 'AMV ' LIST .... 'hdrl'` (§1, FOURCCs at
//!    file offsets `0x00..0x18`).
//! 2. `amvh` body of `0x38` bytes carrying `dwMicroSecPerFrame`, width,
//!    height, fps, and the byte-packed duration (§2).
//! 3. Two `strl` lists (video then audio), each `strh` + `strf`. The
//!    video `strh` / `strf` bodies are all-zero; the audio `strf` is a
//!    20-byte `WAVEFORMATEX` (§3).
//! 4. `movi` payload as a flat alternation of `00dc` (video) / `01wb`
//!    (audio) leaf chunks, **no even-byte alignment** (advance =
//!    `8 + size`), bounded by an `AMV_END_` ASCII trailer (§4).
//!
//! Constants are pulled directly from the doc tables; the layout is
//! position-coded (no length fields to trust for the LIST containers,
//! since the trace shows all RIFF/LIST sizes are zeroed in real
//! files).

use crate::AmvDemuxerError;

/// FORM type FOURCC — the four bytes immediately following the
/// `RIFF <size>` prefix. `b"AMV "` includes the trailing **space**
/// that distinguishes AMV from a conforming AVI file (which carries
/// `b"AVI "`).
pub const AMV_FORM_TYPE: [u8; 4] = *b"AMV ";

/// ASCII trailer that bounds the `movi` payload (§4c). The file ends
/// with this literal sequence immediately after the final `01wb`
/// leaf, taking the place of the `idx1` index that a conforming AVI
/// would carry.
pub const AMV_END_TRAILER: [u8; 8] = *b"AMV_END_";

/// `amvh` body length (§2). The constant `0x38 = 56` is repeated in
/// both observed fixtures and is recorded explicitly here so callers
/// can sanity-check it before deciding to trust the body fields.
pub const AMVH_BODY_LEN: u32 = 0x38;

/// Video leaf chunk tag (§4a). The two-digit stream-index prefix
/// `"00"` plus the AVI convention `dc` for "DIB compressed".
pub const VIDEO_CHUNK_TAG: [u8; 4] = *b"00dc";

/// Audio leaf chunk tag (§4b). The two-digit stream-index prefix
/// `"01"` plus the AVI convention `wb` for "wave bytes".
pub const AUDIO_CHUNK_TAG: [u8; 4] = *b"01wb";

/// Required FOURCCs / list-types in the top-level + `hdrl` walk.
/// Centralised here so the test suite can lean on them directly.
const TAG_RIFF: [u8; 4] = *b"RIFF";
const TAG_LIST: [u8; 4] = *b"LIST";
const TAG_HDRL: [u8; 4] = *b"hdrl";
const TAG_AMVH: [u8; 4] = *b"amvh";
const TAG_STRL: [u8; 4] = *b"strl";
const TAG_STRH: [u8; 4] = *b"strh";
const TAG_STRF: [u8; 4] = *b"strf";
const TAG_MOVI: [u8; 4] = *b"movi";

/// File offset where the top-level `LIST hdrl` block starts (after the
/// 12-byte `RIFF <size> 'AMV '` prefix).
const HDRL_OFFSET: u64 = 0x0C;
/// File offset where the `amvh` FOURCC starts (immediately after
/// `LIST <size> 'hdrl'`).
const AMVH_OFFSET: u64 = 0x18;
/// File offset where the first stream-list (`LIST .. strl`, video)
/// starts. Derived from `0x18 + 8 (FOURCC + body-len) + 0x38 (amvh body)`.
const STRL_VIDEO_OFFSET: u64 = AMVH_OFFSET + 8 + AMVH_BODY_LEN as u64;

/// Video stream-header body length declared by the leaf (`strh`
/// length field). The observed value is `0x38 = 56`; the body itself
/// is all-zero (the device hardcodes the video codec parameters), so
/// we record only its length, not its content.
const STRH_VIDEO_BODY_LEN: u32 = 0x38;

/// Video stream-format body length (`strf` length). Observed `0x24 =
/// 36`; the body is all-zero, no `BITMAPINFOHEADER` is present.
const STRF_VIDEO_BODY_LEN: u32 = 0x24;

/// Audio stream-header body length. Observed `0x30 = 48` and entirely
/// zero in both fixtures (no `auds` / sample-rate / bitrate metadata
/// at the strh level — those live in the audio `strf`).
const STRH_AUDIO_BODY_LEN: u32 = 0x30;

/// Audio stream-format body length. Observed `0x14 = 20`: a
/// `WAVEFORMATEX` with a trailing `cbSize` word plus one pad byte.
const STRF_AUDIO_BODY_LEN: u32 = 0x14;

/// Structured view of the `amvh` main header (§2). All multi-byte
/// integers are little-endian. The seven reserved dwords between
/// `dwMicroSecPerFrame` and `width` are not exposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AmvHeader {
    /// `dwMicroSecPerFrame` (offset 0x00 within the body). Equal to
    /// `1_000_000 / fps`; the two fixtures hold `83_333` (12 fps) and
    /// `62_500` (16 fps).
    pub micros_per_frame: u32,
    /// Video width in pixels (offset 0x20 within the body).
    pub width: u32,
    /// Video height in pixels (offset 0x24 within the body).
    pub height: u32,
    /// Frames per second (offset 0x28 within the body).
    pub fps: u32,
    /// Constant `1` flag at offset 0x2C. Meaning not determinable
    /// from the bytes; surfaced for callers that want to validate it
    /// matches the observed convention.
    pub flag_one: u32,
    /// Reserved dword at offset 0x30. Always zero in observed
    /// fixtures.
    pub reserved_30: u32,
    /// Byte-packed total duration (§2). Encoded as
    /// `[seconds, minutes, hours, 0]` little-endian — see
    /// [`AmvDuration::from_packed`].
    pub duration_packed: u32,
}

impl AmvHeader {
    /// Parse the 56-byte `amvh` body from a slice. Returns
    /// `Error::InvalidData` if the slice is too short.
    pub fn parse(body: &[u8]) -> Result<Self, AmvDemuxerError> {
        if body.len() < AMVH_BODY_LEN as usize {
            return Err(AmvDemuxerError::InvalidData(format!(
                "amvh body must be {} bytes, got {}",
                AMVH_BODY_LEN,
                body.len()
            )));
        }
        Ok(Self {
            micros_per_frame: read_u32_le(body, 0x00),
            width: read_u32_le(body, 0x20),
            height: read_u32_le(body, 0x24),
            fps: read_u32_le(body, 0x28),
            flag_one: read_u32_le(body, 0x2C),
            reserved_30: read_u32_le(body, 0x30),
            duration_packed: read_u32_le(body, 0x34),
        })
    }

    /// Decoded total duration of the stream. Always succeeds because
    /// the packed-byte layout admits any input; the caller is
    /// expected to validate sanity (e.g. seconds < 60, minutes < 60)
    /// if it cares.
    pub fn duration(&self) -> AmvDuration {
        AmvDuration::from_packed(self.duration_packed)
    }

    /// Convenience: total duration expressed in microseconds for
    /// inclusion in [`oxideav_core::Demuxer::duration_micros`].
    pub fn duration_micros(&self) -> i64 {
        let d = self.duration();
        let total_seconds = d.hours as i64 * 3600 + d.minutes as i64 * 60 + d.seconds as i64;
        total_seconds * 1_000_000
    }
}

/// Decoded representation of the `amvh` packed duration (§2). The
/// bytes are laid out `[seconds, minutes, hours, 0]` — each value is a
/// raw little-endian byte, **not** BCD or any other encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AmvDuration {
    /// Seconds component (0..=255 in principle; observed values fit
    /// within 0..60).
    pub seconds: u8,
    /// Minutes component.
    pub minutes: u8,
    /// Hours component.
    pub hours: u8,
}

impl AmvDuration {
    /// Unpack the four bytes of `duration_packed` from `amvh +0x34`.
    pub fn from_packed(packed: u32) -> Self {
        let bytes = packed.to_le_bytes();
        Self {
            seconds: bytes[0],
            minutes: bytes[1],
            hours: bytes[2],
        }
    }
}

/// Decoded view of the audio `WAVEFORMATEX` (§3b). All multi-byte
/// integers little-endian. The header *declares* PCM but, per the
/// trace, the actual audio payload is a 4-bit-per-sample ADPCM-style
/// stream — the [`AmvWaveFormat::format_tag`] field is exposed
/// verbatim so callers can verify the discrepancy themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AmvWaveFormat {
    /// `wFormatTag`. Observed value: `1` (PCM declared; payload is
    /// actually ADPCM — header lies about the codec).
    pub format_tag: u16,
    /// `nChannels`. Always 1 (mono) in observed fixtures.
    pub channels: u16,
    /// `nSamplesPerSec`. Observed values: `22_050`.
    pub samples_per_sec: u32,
    /// `nAvgBytesPerSec`. Observed: `samples_per_sec * 2` — i.e. the
    /// rate of the decoded 16-bit PCM, not the on-disk byte rate.
    pub avg_bytes_per_sec: u32,
    /// `nBlockAlign`. Observed value: 2.
    pub block_align: u16,
    /// `wBitsPerSample`. Observed value: 16 (refers to decoded PCM
    /// width, not the on-disk nibble payload).
    pub bits_per_sample: u16,
    /// `cbSize`. Observed value: 0.
    pub cb_size: u16,
}

impl AmvWaveFormat {
    /// Parse the 20-byte audio `strf` body.
    pub fn parse(body: &[u8]) -> Result<Self, AmvDemuxerError> {
        if body.len() < 18 {
            return Err(AmvDemuxerError::InvalidData(format!(
                "audio strf body must be at least 18 bytes, got {}",
                body.len()
            )));
        }
        Ok(Self {
            format_tag: read_u16_le(body, 0x00),
            channels: read_u16_le(body, 0x02),
            samples_per_sec: read_u32_le(body, 0x04),
            avg_bytes_per_sec: read_u32_le(body, 0x08),
            block_align: read_u16_le(body, 0x0C),
            bits_per_sample: read_u16_le(body, 0x0E),
            cb_size: if body.len() >= 20 {
                read_u16_le(body, 0x10)
            } else {
                0
            },
        })
    }
}

/// Identifies which leaf-chunk tag was seen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkKind {
    /// `00dc` — video frame (JPEG SOI..EOI, tables stripped).
    Video,
    /// `01wb` — audio block (8-byte preamble + ~4-bit ADPCM body).
    Audio,
    /// Anything else encountered inside `movi`. Real AMV files never
    /// produce this in practice (only `00dc` and `01wb` are observed)
    /// but the parser tolerates it so a partial / unexpected chunk
    /// surfaces as data rather than aborting the walk.
    Other([u8; 4]),
}

impl ChunkKind {
    /// Classify a 4-byte FOURCC.
    pub fn classify(tag: [u8; 4]) -> Self {
        if tag == VIDEO_CHUNK_TAG {
            Self::Video
        } else if tag == AUDIO_CHUNK_TAG {
            Self::Audio
        } else {
            Self::Other(tag)
        }
    }
}

/// 8-byte leaf-chunk header (`FOURCC` + 4-byte little-endian size).
/// **Critically**, AMV does NOT pad chunks to an even byte boundary —
/// consumers walk by exactly `8 + size` bytes per chunk, even when
/// `size` is odd (§4 "Chunk framing and the no-padding rule").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkHeader {
    /// Raw 4-byte tag (`b"00dc"` / `b"01wb"` / …).
    pub tag: [u8; 4],
    /// Body size in bytes. Excludes this 8-byte header.
    pub size: u32,
}

impl ChunkHeader {
    /// Parse an 8-byte chunk header from the start of `slice`.
    pub fn parse(slice: &[u8]) -> Result<Self, AmvDemuxerError> {
        if slice.len() < 8 {
            return Err(AmvDemuxerError::InvalidData(format!(
                "chunk header needs 8 bytes, got {}",
                slice.len()
            )));
        }
        let mut tag = [0u8; 4];
        tag.copy_from_slice(&slice[0..4]);
        let size = read_u32_le(slice, 4);
        Ok(Self { tag, size })
    }

    /// Number of bytes the cursor must advance to land on the next
    /// chunk: exactly `8 + size` (no even-byte padding — §4).
    pub fn advance_total(&self) -> u64 {
        8 + self.size as u64
    }

    /// Classify the chunk tag.
    pub fn kind(&self) -> ChunkKind {
        ChunkKind::classify(self.tag)
    }
}

// ────────────────────────── prelude walker ──────────────────────────

/// Parsed view of the AMV prelude — everything from offset 0 up to the
/// start of the `movi` payload. Used by [`crate::AmvDemuxer::open`] so
/// the demuxer can pick out the streams without doing the I/O walk in
/// its own constructor.
#[derive(Clone, Debug)]
pub(crate) struct AmvPrelude {
    /// Decoded `amvh` body.
    pub header: AmvHeader,
    /// Decoded audio `WAVEFORMATEX` from the audio `strf` leaf.
    pub audio_format: AmvWaveFormat,
    /// File offset of the first leaf chunk inside `movi` (the byte
    /// **after** the `movi` FOURCC). Walking starts here.
    pub movi_payload_start: u64,
}

impl AmvPrelude {
    /// Parse the file prelude from a contiguous byte slice that begins
    /// at file offset 0 and contains at least up to and including the
    /// `LIST <size> 'movi'` opener. The returned
    /// `movi_payload_start` is an absolute file offset.
    pub fn parse(slice: &[u8]) -> Result<Self, AmvDemuxerError> {
        // 1. Top-level: RIFF <size> 'AMV ' LIST <size> 'hdrl'.
        require_tag(slice, 0x00, TAG_RIFF, "top-level RIFF")?;
        require_tag(slice, 0x08, AMV_FORM_TYPE, "FORM type 'AMV '")?;
        require_tag(slice, HDRL_OFFSET as usize, TAG_LIST, "hdrl LIST opener")?;
        require_tag(
            slice,
            HDRL_OFFSET as usize + 8,
            TAG_HDRL,
            "hdrl list-type tag",
        )?;

        // 2. `amvh` leaf at AMVH_OFFSET.
        require_tag(slice, AMVH_OFFSET as usize, TAG_AMVH, "amvh leaf")?;
        let amvh_size = read_u32_le(slice, AMVH_OFFSET as usize + 4);
        if amvh_size != AMVH_BODY_LEN {
            return Err(AmvDemuxerError::InvalidData(format!(
                "amvh body length must be {AMVH_BODY_LEN}, got {amvh_size}"
            )));
        }
        let amvh_body_start = AMVH_OFFSET as usize + 8;
        let amvh_body_end = amvh_body_start + AMVH_BODY_LEN as usize;
        if slice.len() < amvh_body_end {
            return Err(AmvDemuxerError::InvalidData(
                "prelude slice shorter than amvh body".into(),
            ));
        }
        let header = AmvHeader::parse(&slice[amvh_body_start..amvh_body_end])?;

        // 3a. Video strl (offsets per trace doc §3a):
        //   STRL_VIDEO_OFFSET    = LIST
        //   +0x08                = 'strl'
        //   +0x0C                = 'strh', size = 0x38
        //   +0x14                = strh body (56 bytes, all zero)
        //   +0x4C                = 'strf', size = 0x24
        //   +0x54                = strf body (36 bytes, all zero)
        //   +0x78                = end of video strl
        let v = STRL_VIDEO_OFFSET as usize;
        require_tag(slice, v, TAG_LIST, "video strl LIST")?;
        require_tag(slice, v + 8, TAG_STRL, "video strl type tag")?;
        require_tag(slice, v + 12, TAG_STRH, "video strh leaf")?;
        let v_strh_size = read_u32_le(slice, v + 16);
        if v_strh_size != STRH_VIDEO_BODY_LEN {
            return Err(AmvDemuxerError::InvalidData(format!(
                "video strh body length must be {STRH_VIDEO_BODY_LEN}, got {v_strh_size}"
            )));
        }
        let v_strf_at = v + 20 + STRH_VIDEO_BODY_LEN as usize;
        require_tag(slice, v_strf_at, TAG_STRF, "video strf leaf")?;
        let v_strf_size = read_u32_le(slice, v_strf_at + 4);
        if v_strf_size != STRF_VIDEO_BODY_LEN {
            return Err(AmvDemuxerError::InvalidData(format!(
                "video strf body length must be {STRF_VIDEO_BODY_LEN}, got {v_strf_size}"
            )));
        }
        // 3b. Audio strl — immediately follows the video strl.
        let a = v_strf_at + 8 + STRF_VIDEO_BODY_LEN as usize;
        require_tag(slice, a, TAG_LIST, "audio strl LIST")?;
        require_tag(slice, a + 8, TAG_STRL, "audio strl type tag")?;
        require_tag(slice, a + 12, TAG_STRH, "audio strh leaf")?;
        let a_strh_size = read_u32_le(slice, a + 16);
        if a_strh_size != STRH_AUDIO_BODY_LEN {
            return Err(AmvDemuxerError::InvalidData(format!(
                "audio strh body length must be {STRH_AUDIO_BODY_LEN}, got {a_strh_size}"
            )));
        }
        let a_strf_at = a + 20 + STRH_AUDIO_BODY_LEN as usize;
        require_tag(slice, a_strf_at, TAG_STRF, "audio strf leaf")?;
        let a_strf_size = read_u32_le(slice, a_strf_at + 4);
        if a_strf_size != STRF_AUDIO_BODY_LEN {
            return Err(AmvDemuxerError::InvalidData(format!(
                "audio strf body length must be {STRF_AUDIO_BODY_LEN}, got {a_strf_size}"
            )));
        }
        let a_strf_body_start = a_strf_at + 8;
        let a_strf_body_end = a_strf_body_start + STRF_AUDIO_BODY_LEN as usize;
        if slice.len() < a_strf_body_end {
            return Err(AmvDemuxerError::InvalidData(
                "prelude slice shorter than audio strf body".into(),
            ));
        }
        let audio_format = AmvWaveFormat::parse(&slice[a_strf_body_start..a_strf_body_end])?;

        // 4. `LIST <size> 'movi'` opener — immediately after the
        //    audio strl. `movi` payload starts 4 bytes after the
        //    `movi` FOURCC (LIST size precedes the type tag).
        let movi_list = a_strf_body_end;
        require_tag(slice, movi_list, TAG_LIST, "movi LIST opener")?;
        require_tag(slice, movi_list + 8, TAG_MOVI, "movi list-type tag")?;
        let movi_payload_start = (movi_list + 12) as u64;

        Ok(Self {
            header,
            audio_format,
            movi_payload_start,
        })
    }
}

/// Minimum number of bytes the demuxer reads up-front to parse the
/// whole prelude. Computed from the fixed offsets above; the
/// `comedian.amv` fixture's `movi` payload starts at 0x13C, this
/// constant matches.
pub(crate) const PRELUDE_MIN_LEN: usize = 0x13C;

// ────────────────────────── helpers ──────────────────────────

fn read_u16_le(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([buf[off], buf[off + 1]])
}

fn read_u32_le(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn require_tag(
    slice: &[u8],
    at: usize,
    expected: [u8; 4],
    label: &str,
) -> Result<(), AmvDemuxerError> {
    if slice.len() < at + 4 {
        return Err(AmvDemuxerError::InvalidData(format!(
            "slice too short for {label} at offset {at:#x}"
        )));
    }
    let mut got = [0u8; 4];
    got.copy_from_slice(&slice[at..at + 4]);
    if got != expected {
        return Err(AmvDemuxerError::InvalidData(format!(
            "{label}: expected FOURCC {:?}, got {:?} at offset {at:#x}",
            std::str::from_utf8(&expected).unwrap_or("?"),
            std::str::from_utf8(&got).unwrap_or("?")
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn duration_unpacks_comedian_value() {
        // §2: `21 01 00 00` → 0x21 = 33 s, 0x01 = 1 min → 1:33.
        let d = AmvDuration::from_packed(0x0000_0121);
        assert_eq!(d.seconds, 0x21);
        assert_eq!(d.minutes, 0x01);
        assert_eq!(d.hours, 0x00);
    }

    #[test]
    fn duration_unpacks_noel_value() {
        // §2: `02 03 00 00` → 0x02 = 2 s, 0x03 = 3 min → 3:02.
        let d = AmvDuration::from_packed(0x0000_0302);
        assert_eq!(d.seconds, 2);
        assert_eq!(d.minutes, 3);
        assert_eq!(d.hours, 0);
    }

    #[test]
    fn amvh_parse_comedian_body() {
        // Manually assemble the 56-byte body for comedian.amv as
        // documented in §2: micros_per_frame = 83333, width = 128,
        // height = 96, fps = 12, flag_one = 1, duration_packed = 0x0121.
        let mut body = vec![0u8; AMVH_BODY_LEN as usize];
        body[0..4].copy_from_slice(&83_333u32.to_le_bytes());
        body[0x20..0x24].copy_from_slice(&128u32.to_le_bytes());
        body[0x24..0x28].copy_from_slice(&96u32.to_le_bytes());
        body[0x28..0x2C].copy_from_slice(&12u32.to_le_bytes());
        body[0x2C..0x30].copy_from_slice(&1u32.to_le_bytes());
        body[0x34..0x38].copy_from_slice(&0x0000_0121u32.to_le_bytes());
        let h = AmvHeader::parse(&body).unwrap();
        assert_eq!(h.micros_per_frame, 83_333);
        assert_eq!(h.width, 128);
        assert_eq!(h.height, 96);
        assert_eq!(h.fps, 12);
        assert_eq!(h.flag_one, 1);
        assert_eq!(
            h.duration(),
            AmvDuration {
                seconds: 0x21,
                minutes: 1,
                hours: 0
            }
        );
        // Total micros = (1 * 60 + 33) * 1_000_000 = 93_000_000.
        assert_eq!(h.duration_micros(), 93_000_000);
    }

    #[test]
    fn amvh_parse_rejects_short_body() {
        let body = vec![0u8; 40];
        assert!(AmvHeader::parse(&body).is_err());
    }

    #[test]
    fn waveformat_parse_comedian_strf() {
        // §3b WAVEFORMATEX: tag=1, ch=1, 22050, 44100, blockAlign=2,
        // bps=16, cbSize=0. 20 bytes total.
        let mut body = vec![0u8; STRF_AUDIO_BODY_LEN as usize];
        body[0..2].copy_from_slice(&1u16.to_le_bytes());
        body[2..4].copy_from_slice(&1u16.to_le_bytes());
        body[4..8].copy_from_slice(&22_050u32.to_le_bytes());
        body[8..12].copy_from_slice(&44_100u32.to_le_bytes());
        body[12..14].copy_from_slice(&2u16.to_le_bytes());
        body[14..16].copy_from_slice(&16u16.to_le_bytes());
        body[16..18].copy_from_slice(&0u16.to_le_bytes());
        let fmt = AmvWaveFormat::parse(&body).unwrap();
        assert_eq!(fmt.format_tag, 1);
        assert_eq!(fmt.channels, 1);
        assert_eq!(fmt.samples_per_sec, 22_050);
        assert_eq!(fmt.avg_bytes_per_sec, 44_100);
        assert_eq!(fmt.block_align, 2);
        assert_eq!(fmt.bits_per_sample, 16);
        assert_eq!(fmt.cb_size, 0);
    }

    #[test]
    fn chunk_kind_classifies_known_tags() {
        assert_eq!(ChunkKind::classify(VIDEO_CHUNK_TAG), ChunkKind::Video);
        assert_eq!(ChunkKind::classify(AUDIO_CHUNK_TAG), ChunkKind::Audio);
        assert_eq!(ChunkKind::classify(*b"abcd"), ChunkKind::Other(*b"abcd"));
    }

    #[test]
    fn chunk_header_advance_total_is_no_padding() {
        // §4 "no even-byte alignment" — advance = 8 + size exactly,
        // even for odd-sized payloads.
        let h = ChunkHeader {
            tag: VIDEO_CHUNK_TAG,
            size: 1633,
        };
        assert_eq!(h.advance_total(), 1641);
        let h_odd = ChunkHeader {
            tag: AUDIO_CHUNK_TAG,
            size: 927,
        };
        assert_eq!(h_odd.advance_total(), 935);
    }

    #[test]
    fn prelude_parse_accepts_synthetic_minimal_amv() {
        // Build a synthetic 12-byte-header AMV prelude with exactly
        // the byte layout the trace doc documents. Verify the
        // strict-position parser accepts it and surfaces the right
        // amvh / strf values.
        let buf = build_synthetic_prelude(128, 96, 12, 0x0000_0121, 22_050);
        let prelude = AmvPrelude::parse(&buf).expect("prelude parses");
        assert_eq!(prelude.header.width, 128);
        assert_eq!(prelude.header.height, 96);
        assert_eq!(prelude.header.fps, 12);
        assert_eq!(prelude.header.micros_per_frame, 83_333);
        assert_eq!(prelude.audio_format.samples_per_sec, 22_050);
        assert_eq!(prelude.movi_payload_start, 0x13C);
    }

    #[test]
    fn prelude_parse_rejects_avi_form_type() {
        let mut buf = build_synthetic_prelude(128, 96, 12, 0x0000_0121, 22_050);
        // Overwrite FORM type 'AMV ' with 'AVI '.
        buf[8..12].copy_from_slice(b"AVI ");
        assert!(AmvPrelude::parse(&buf).is_err());
    }

    /// Helper: assemble a minimal but byte-correct AMV prelude (from
    /// offset 0 up through the `movi` opener) for testing.
    ///
    /// `duration_packed` is the raw u32 to write at `amvh +0x34`. Use
    /// [`crate::AmvDuration::from_packed`] to construct sample
    /// values.
    pub(crate) fn build_synthetic_prelude(
        width: u32,
        height: u32,
        fps: u32,
        duration_packed: u32,
        samples_per_sec: u32,
    ) -> Vec<u8> {
        let mut buf = Vec::with_capacity(0x140);
        // RIFF + zeroed size + 'AMV '.
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(b"AMV ");
        // hdrl LIST opener.
        buf.extend_from_slice(b"LIST");
        buf.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(b"hdrl");
        // amvh leaf: FOURCC + size + 56-byte body.
        buf.extend_from_slice(b"amvh");
        buf.extend_from_slice(&AMVH_BODY_LEN.to_le_bytes());
        let mut amvh_body = vec![0u8; AMVH_BODY_LEN as usize];
        let micros = 1_000_000 / fps;
        amvh_body[0..4].copy_from_slice(&micros.to_le_bytes());
        amvh_body[0x20..0x24].copy_from_slice(&width.to_le_bytes());
        amvh_body[0x24..0x28].copy_from_slice(&height.to_le_bytes());
        amvh_body[0x28..0x2C].copy_from_slice(&fps.to_le_bytes());
        amvh_body[0x2C..0x30].copy_from_slice(&1u32.to_le_bytes());
        amvh_body[0x34..0x38].copy_from_slice(&duration_packed.to_le_bytes());
        buf.extend_from_slice(&amvh_body);
        // Video strl: LIST <size=0> strl strh <0x38, all-zero> strf <0x24, all-zero>
        buf.extend_from_slice(b"LIST");
        buf.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(b"strl");
        buf.extend_from_slice(b"strh");
        buf.extend_from_slice(&STRH_VIDEO_BODY_LEN.to_le_bytes());
        buf.extend_from_slice(&vec![0u8; STRH_VIDEO_BODY_LEN as usize]);
        buf.extend_from_slice(b"strf");
        buf.extend_from_slice(&STRF_VIDEO_BODY_LEN.to_le_bytes());
        buf.extend_from_slice(&vec![0u8; STRF_VIDEO_BODY_LEN as usize]);
        // Audio strl: LIST <0> strl strh <0x30, all-zero> strf <0x14, WAVEFORMATEX>.
        buf.extend_from_slice(b"LIST");
        buf.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(b"strl");
        buf.extend_from_slice(b"strh");
        buf.extend_from_slice(&STRH_AUDIO_BODY_LEN.to_le_bytes());
        buf.extend_from_slice(&vec![0u8; STRH_AUDIO_BODY_LEN as usize]);
        buf.extend_from_slice(b"strf");
        buf.extend_from_slice(&STRF_AUDIO_BODY_LEN.to_le_bytes());
        let mut strf_audio = vec![0u8; STRF_AUDIO_BODY_LEN as usize];
        strf_audio[0..2].copy_from_slice(&1u16.to_le_bytes());
        strf_audio[2..4].copy_from_slice(&1u16.to_le_bytes());
        strf_audio[4..8].copy_from_slice(&samples_per_sec.to_le_bytes());
        strf_audio[8..12].copy_from_slice(&(samples_per_sec * 2).to_le_bytes());
        strf_audio[12..14].copy_from_slice(&2u16.to_le_bytes());
        strf_audio[14..16].copy_from_slice(&16u16.to_le_bytes());
        strf_audio[16..18].copy_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&strf_audio);
        // movi LIST opener.
        buf.extend_from_slice(b"LIST");
        buf.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(b"movi");
        assert_eq!(buf.len(), PRELUDE_MIN_LEN);
        buf
    }
}
