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

    /// Strict sentinel validation of the parsed §2 `amvh` body, applying
    /// the cross-checks that the byte-level [`Self::parse`] permissively
    /// skips. Per trace §2 the body fixes three relationships that a
    /// conforming device-profile file always satisfies:
    ///
    /// * `dwMicroSecPerFrame == 1_000_000 / fps` (the `+0x00` derivation
    ///   from `+0x28`). Both observed fixtures match — comedian holds
    ///   `83_333` for 12 fps, noel holds `62_500` for 16 fps. A
    ///   contradicting value indicates a corrupted header.
    /// * `flag_one == 1` (the `+0x2C` constant). Both observed fixtures
    ///   carry `1`; the trace records this as a constant of unknown
    ///   meaning but stable value, so strict callers treat any other
    ///   value as a signal that the header is not a real AMV header.
    /// * `reserved_30 == 0` (the `+0x30` reserved dword). Always zero in
    ///   both fixtures per §2's "reserved / zeroed" annotation.
    ///
    /// The permissive [`Self::parse`] path stays untouched so the
    /// existing demuxer-open path continues to accept any byte-shaped
    /// `amvh` body; this method is the opt-in cross-check for callers
    /// that want to reject non-conforming files up-front.
    ///
    /// The `fps == 0` corner — for which the `1_000_000 / fps` derivation
    /// is undefined — is reported as a separate `fps must be > 0` error
    /// rather than triggering a division by zero.
    pub fn validate_sentinels(&self) -> Result<(), AmvDemuxerError> {
        if self.fps == 0 {
            return Err(AmvDemuxerError::InvalidData(
                "amvh +0x28 fps must be > 0".into(),
            ));
        }
        let expected_micros = 1_000_000 / self.fps;
        if self.micros_per_frame != expected_micros {
            return Err(AmvDemuxerError::InvalidData(format!(
                "amvh +0x00 dwMicroSecPerFrame={} does not match 1_000_000/fps={}",
                self.micros_per_frame, expected_micros
            )));
        }
        if self.flag_one != 1 {
            return Err(AmvDemuxerError::InvalidData(format!(
                "amvh +0x2C constant must be 1, got {}",
                self.flag_one
            )));
        }
        if self.reserved_30 != 0 {
            return Err(AmvDemuxerError::InvalidData(format!(
                "amvh +0x30 reserved dword must be 0, got {}",
                self.reserved_30
            )));
        }
        Ok(())
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

    /// Inverse of [`Self::from_packed`]: pack the
    /// `[seconds, minutes, hours, 0]` byte layout back into the raw
    /// little-endian `u32` written at `amvh +0x34` (§2).
    ///
    /// The fourth byte is always written as `0` per the trace doc — the
    /// two observed device profiles both leave it zero (`21 01 00 00`
    /// for the 12 fps profile, `02 03 00 00` for the 16 fps profile),
    /// and the trace records the field as `[seconds, minutes, hours, 0]`.
    ///
    /// Useful for tooling that wants to recompute or patch the duration
    /// field independently of the muxer's `write_trailer` path — for
    /// example, when re-stamping a recovered truncated file's header
    /// after the chunk count has been determined.
    pub fn to_packed(&self) -> u32 {
        u32::from_le_bytes([self.seconds, self.minutes, self.hours, 0])
    }

    /// Total duration expressed in whole seconds. Saturates at
    /// [`u32::MAX`] to keep the conversion infallible — the field is
    /// 8-bit-per-component so the maximum representable duration is
    /// `255 h 255 min 255 s = 933 555 s`, which fits comfortably.
    ///
    /// Provided as a convenience for tooling that wants the same
    /// derivation [`AmvHeader::duration_micros`] performs without going
    /// through the µs detour.
    pub fn total_seconds(&self) -> u32 {
        self.hours as u32 * 3600 + self.minutes as u32 * 60 + self.seconds as u32
    }

    /// Derive an `AmvDuration` from an observed video-chunk count and
    /// the §2 `fps` field, applying the worked example the trace records
    /// for `comedian.amv`: "1116 frames ÷ 12 fps = 93 s = 1:33".
    ///
    /// Implementation: `total_seconds = frame_count / fps`, then break
    /// that out into `[seconds, minutes, hours]` using whole-minute /
    /// whole-hour division so the result re-packs (via
    /// [`Self::to_packed`]) into the same little-endian dword the
    /// device writes at `amvh +0x34` — `0x0000_0121` for the comedian
    /// profile, `0x0000_0302` for the noel profile. Components saturate
    /// at [`u8::MAX`] to keep the function infallible; the field's
    /// per-component byte width means the maximum representable
    /// duration is `255 h 59 min 59 s = 921 599 s`, well above the
    /// two observed fixtures (93 s and 182 s).
    ///
    /// Returns the all-zero duration (`seconds = minutes = hours = 0`)
    /// when `fps == 0` — the trace records `fps > 0` for every device
    /// profile observed, so a zero rate is rejected at higher layers
    /// (e.g. [`AmvHeader::validate_sentinels`]); this guard exists only
    /// so the helper itself stays division-by-zero-safe.
    ///
    /// Useful for tooling that wants to recompute the `amvh +0x34`
    /// packed-byte duration independently of the muxer's
    /// `write_trailer` patch path — for example, when re-stamping a
    /// recovered truncated file's header after the surviving chunk
    /// count has been determined.
    pub fn from_frame_count(frame_count: u64, fps: u32) -> Self {
        if fps == 0 {
            return Self {
                seconds: 0,
                minutes: 0,
                hours: 0,
            };
        }
        let total_seconds = frame_count / fps as u64;
        let hours = (total_seconds / 3600).min(u8::MAX as u64) as u8;
        let after_hours = total_seconds.saturating_sub(hours as u64 * 3600);
        let minutes = (after_hours / 60).min(u8::MAX as u64) as u8;
        let seconds = (after_hours % 60).min(u8::MAX as u64) as u8;
        Self {
            seconds,
            minutes,
            hours,
        }
    }

    /// Cross-check the parsed `[seconds, minutes, hours]` triple against
    /// an observed video-chunk count and the §2 `fps` field, using the
    /// same derivation the trace's worked example applies to the
    /// comedian fixture: "1116 frames ÷ 12 fps = 93 s = 1:33". Returns
    /// `true` when `self == AmvDuration::from_frame_count(frame_count,
    /// fps)` and `false` otherwise.
    ///
    /// Provided so tooling that has both a parsed `amvh` header and a
    /// completed `movi` walk (e.g. after a truncation-recovery pass)
    /// can confirm the header's packed duration is consistent with the
    /// chunks that actually landed without re-implementing the
    /// derivation. Returns `false` when `fps == 0` unless `self` is
    /// also the all-zero duration, mirroring [`Self::from_frame_count`]'s
    /// zero-fps guard.
    pub fn is_consistent_with_frame_count(&self, frame_count: u64, fps: u32) -> bool {
        *self == Self::from_frame_count(frame_count, fps)
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

    /// Strict sentinel validation of the parsed §3b audio
    /// `WAVEFORMATEX` body, applying the cross-checks that the
    /// permissive [`Self::parse`] path skips. Per trace §3b the device
    /// profile fixes six relationships:
    ///
    /// * `wFormatTag == 1` (the `+0x00` declared-PCM tag — the payload
    ///   is actually ADPCM-style but the header advertises 0x0001).
    /// * `nChannels == 1` (the `+0x02` mono channel count — observed
    ///   identically in both fixtures).
    /// * `nAvgBytesPerSec == nSamplesPerSec * 2` (the `+0x08`
    ///   derivation from `+0x04`, describing the **decoded** 16-bit PCM
    ///   rate; observed `44_100 == 22_050 * 2`).
    /// * `nBlockAlign == 2` (the `+0x0C` two-byte block alignment of
    ///   the declared 16-bit PCM stream).
    /// * `wBitsPerSample == 16` (the `+0x0E` decoded-sample-width — the
    ///   on-disk payload is `~4` bits/sample but the header declares
    ///   the decoded-PCM width).
    /// * `cbSize == 0` (the `+0x10` `WAVEFORMATEX` extension-size
    ///   marker; observed zero in both fixtures since the device emits
    ///   no `WAVEFORMATEXTENSIBLE` tail).
    ///
    /// `samples_per_sec` itself is left unvalidated — the trace records
    /// only the one observation (`22_050`) and does not document
    /// whether other Actions / ALi-chip device profiles also vary the
    /// audio sample rate, so strict callers do not gate on it.
    ///
    /// The permissive [`Self::parse`] path stays untouched so the
    /// existing demuxer-open path continues to accept any byte-shaped
    /// 18+ byte audio `strf` body; this method is the opt-in
    /// cross-check for callers that want device-profile strictness.
    pub fn validate_sentinels(&self) -> Result<(), AmvDemuxerError> {
        if self.format_tag != 1 {
            return Err(AmvDemuxerError::InvalidData(format!(
                "audio strf +0x00 wFormatTag must be 1 (declared PCM), got {}",
                self.format_tag
            )));
        }
        if self.channels != 1 {
            return Err(AmvDemuxerError::InvalidData(format!(
                "audio strf +0x02 nChannels must be 1 (mono), got {}",
                self.channels
            )));
        }
        let expected_avg = self.samples_per_sec.saturating_mul(2);
        if self.avg_bytes_per_sec != expected_avg {
            return Err(AmvDemuxerError::InvalidData(format!(
                "audio strf +0x08 nAvgBytesPerSec={} does not match \
                 nSamplesPerSec * 2 = {}",
                self.avg_bytes_per_sec, expected_avg
            )));
        }
        if self.block_align != 2 {
            return Err(AmvDemuxerError::InvalidData(format!(
                "audio strf +0x0C nBlockAlign must be 2, got {}",
                self.block_align
            )));
        }
        if self.bits_per_sample != 16 {
            return Err(AmvDemuxerError::InvalidData(format!(
                "audio strf +0x0E wBitsPerSample must be 16, got {}",
                self.bits_per_sample
            )));
        }
        if self.cb_size != 0 {
            return Err(AmvDemuxerError::InvalidData(format!(
                "audio strf +0x10 cbSize must be 0, got {}",
                self.cb_size
            )));
        }
        Ok(())
    }

    /// Per-frame-interval audio sample budget the trace's §4b worked
    /// example records — `nSamplesPerSec ÷ fps`, integer truncation.
    ///
    /// The trace records the §4b first audio block of `comedian.amv`
    /// carries `decoded_sample_count = 1837` and notes that this matches
    /// `22 050 Hz ÷ 12 fps ≈ 1837` mono samples — i.e. each `01wb` block
    /// holds exactly one video-frame-interval worth of audio under the
    /// §4 strict 1:1 video-first interleave rule. This helper exposes
    /// that derivation as a pure function so tooling that has already
    /// parsed the audio `WAVEFORMATEX` (§3b) and the §2 `fps` can
    /// reproduce the budget without re-implementing the arithmetic, and
    /// so [`AmvAudioPreamble::is_consistent_with_frame_interval`] can
    /// share the exact same computation.
    ///
    /// Returns `0` when `fps == 0` to keep the helper division-by-zero
    /// safe; the trace records `fps > 0` for every device profile
    /// observed, so the zero rate is rejected upstream by
    /// [`AmvHeader::validate_sentinels`].
    pub fn frame_interval_samples(&self, fps: u32) -> u32 {
        if fps == 0 {
            return 0;
        }
        self.samples_per_sec / fps
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

/// JPEG Start-of-Image marker — the two bytes every `00dc` video chunk
/// payload begins with (§4a "self-contained JPEG bracketed by SOI…EOI").
pub const JPEG_SOI: [u8; 2] = [0xFF, 0xD8];

/// JPEG End-of-Image marker — the two bytes every `00dc` video chunk
/// payload ends with (§4a). The trace records both observed device
/// profiles' first frames hold SOI at offset 0 and EOI at `size - 2`.
pub const JPEG_EOI: [u8; 2] = [0xFF, 0xD9];

/// Minimum size of an `01wb` audio chunk payload — the 8-byte §4b
/// preamble alone, with no compressed body. Real chunks are always
/// larger (the comedian profile's blocks are 927 bytes) but this
/// minimum is the smallest payload that satisfies the §4b structural
/// invariant.
pub const AMV_AUDIO_PREAMBLE_LEN: usize = 8;

/// Decoded view of the 8-byte preamble that prefixes every `01wb`
/// audio chunk payload (§4b). Both fields are little-endian `u32`s.
///
/// The trace doc records two observations about this preamble in the
/// `comedian.amv` fixture's first audio block:
///
/// * `state == 0` in the very first block (consistent with "presumably
///   the initial predictor / step index in the first dword"); later
///   blocks carry non-zero state and the trace records no further
///   constants beyond "the per-block state" so the byte parser surfaces
///   the value verbatim.
/// * `decoded_sample_count` matches `nSamplesPerSec ÷ fps` per block
///   (`22_050 ÷ 12 ≈ 1837` mono samples in the comedian first block).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AmvAudioPreamble {
    /// First u32 at preamble +0x00. Per-block state — initial predictor
    /// / step index for the nibble decoder. Surfaced verbatim because
    /// the trace records only one observation (the first block's `0`)
    /// and explicitly notes the field's content varies block-to-block.
    pub state: u32,
    /// Second u32 at preamble +0x04. Number of decoded mono samples the
    /// following compressed body unpacks into. Drives the per-packet
    /// `pts` / `duration` accounting in the demuxer.
    pub decoded_sample_count: u32,
}

impl AmvAudioPreamble {
    /// Parse the 8-byte preamble from the start of an `01wb` payload.
    /// Returns [`AmvDemuxerError::InvalidData`] if the slice is shorter
    /// than [`AMV_AUDIO_PREAMBLE_LEN`].
    pub fn parse(body: &[u8]) -> Result<Self, AmvDemuxerError> {
        if body.len() < AMV_AUDIO_PREAMBLE_LEN {
            return Err(AmvDemuxerError::InvalidData(format!(
                "01wb payload preamble needs {AMV_AUDIO_PREAMBLE_LEN} bytes, got {}",
                body.len()
            )));
        }
        Ok(Self {
            state: read_u32_le(body, 0x00),
            decoded_sample_count: read_u32_le(body, 0x04),
        })
    }

    /// Strict sentinel validation of the parsed §4b preamble. Confirms
    /// the one cross-checkable invariant the trace records:
    ///
    /// * `decoded_sample_count > 0` — every observed audio block in the
    ///   two staged fixtures carries a positive sample count (the
    ///   comedian first block holds `1837 = 22_050 ÷ 12`). A zero count
    ///   would imply an empty block, which neither device profile
    ///   produces.
    ///
    /// The `state` field is intentionally **not** validated: the trace
    /// records only the first-block observation (`0`) and explicitly
    /// flags the field as per-block varying state, so strict callers
    /// cannot gate on it.
    pub fn validate_sentinels(&self) -> Result<(), AmvDemuxerError> {
        if self.decoded_sample_count == 0 {
            return Err(AmvDemuxerError::InvalidData(
                "01wb +0x04 decoded_sample_count must be > 0".into(),
            ));
        }
        Ok(())
    }

    /// Cross-check the parsed preamble's `decoded_sample_count` against
    /// the §4b worked-example frame-interval sample budget derived from
    /// the §3b audio `WAVEFORMATEX` `nSamplesPerSec` and the §2 `fps`.
    ///
    /// Per the trace's §4b worked example, the comedian device profile's
    /// first audio block carries `decoded_sample_count = 1837 =
    /// 22_050 ÷ 12` — each `01wb` block holds exactly one
    /// video-frame-interval of audio under the §4 strict 1:1 video-first
    /// interleave rule. This helper returns `true` when the parsed
    /// `decoded_sample_count` matches that integer-division budget
    /// exactly (i.e. `samples_per_sec / fps`) and `false` otherwise.
    ///
    /// The integer-truncation comparison reflects the trace's exact
    /// recorded value (`1837`, not `1837.5`); callers that want a
    /// tolerance for the occasional `+1` block can compare
    /// `decoded_sample_count.abs_diff(format.frame_interval_samples(fps))`
    /// against their own threshold instead.
    ///
    /// Returns `false` when `fps == 0` (matching the
    /// [`AmvWaveFormat::frame_interval_samples`] zero-fps guard, which
    /// short-circuits to `0`) unless the parsed sample count is also
    /// `0` — which on its own is already rejected by
    /// [`Self::validate_sentinels`].
    ///
    /// Useful for tooling that wants to confirm a per-block sample count
    /// is consistent with the stream's frame-interval budget without
    /// re-implementing the §4b derivation — for example, to flag a
    /// recovered truncated chunk whose preamble was clipped mid-write.
    pub fn is_consistent_with_frame_interval(&self, samples_per_sec: u32, fps: u32) -> bool {
        if fps == 0 {
            return self.decoded_sample_count == 0;
        }
        self.decoded_sample_count == samples_per_sec / fps
    }

    /// Expected compressed-body byte count for this block under the §4b
    /// 4-bit-per-sample nibble-packing relation the trace records.
    ///
    /// Per §4b the `comedian.amv` first audio block carries
    /// `decoded_sample_count = 1837` and a compressed body of `919`
    /// bytes — "1837 mono samples encoded in 919 bytes ≈ 0.5 byte/sample
    /// = 4 bits/sample", an IMA/DVI-ADPCM-style nibble codec where each
    /// mono sample occupies one 4-bit nibble. Two nibbles pack into one
    /// byte, so a block of `n` samples needs `ceil(n / 2)` body bytes —
    /// `ceil(1837 / 2) = 919`, matching the trace's recorded body length
    /// exactly.
    ///
    /// The result is the compressed-body byte count **excluding** the
    /// 8-byte preamble itself (the preamble is the per-block header, not
    /// part of the nibble-coded body). Returned as a `u64` so callers can
    /// compare it against a chunk size without an intermediate cast.
    ///
    /// Useful for tooling that wants to derive the expected nibble-body
    /// length from a parsed preamble — for example, to size a decode
    /// buffer ahead of the unpack pass, or to cross-check a recovered
    /// block's on-disk length against its declared sample count (see
    /// [`Self::is_consistent_with_body_len`]).
    pub fn nibble_body_len(&self) -> u64 {
        (self.decoded_sample_count as u64).div_ceil(2)
    }

    /// Cross-check a full `01wb` payload length against this block's §4b
    /// nibble-packing budget.
    ///
    /// The `total_payload_len` argument is the **complete** `01wb` chunk
    /// payload length — the 8-byte preamble [`AMV_AUDIO_PREAMBLE_LEN`]
    /// plus the compressed nibble body. This helper returns `true` when
    ///
    /// ```text
    /// total_payload_len == AMV_AUDIO_PREAMBLE_LEN + ceil(decoded_sample_count / 2)
    /// ```
    ///
    /// i.e. when the on-disk block length exactly matches the §4b
    /// 4-bit-per-sample relation ([`Self::nibble_body_len`] plus the
    /// preamble), and `false` otherwise. For the `comedian.amv` first
    /// block this is `8 + 919 = 927`, the trace's recorded chunk size.
    ///
    /// A `total_payload_len` shorter than the 8-byte preamble returns
    /// `false` rather than panicking — a sub-preamble length cannot carry
    /// a valid block under §4b regardless of the declared sample count.
    ///
    /// Useful for a truncation-recovery / sanity pass that wants to flag
    /// an `01wb` block whose declared `decoded_sample_count` is
    /// inconsistent with the bytes that actually landed — e.g. a block
    /// clipped mid-write whose compressed body is short of the nibble
    /// budget its preamble promises.
    pub fn is_consistent_with_body_len(&self, total_payload_len: u64) -> bool {
        let preamble = AMV_AUDIO_PREAMBLE_LEN as u64;
        match total_payload_len.checked_sub(preamble) {
            Some(body_len) => body_len == self.nibble_body_len(),
            None => false,
        }
    }

    /// Number of bytes a full `01wb` payload carries **beyond** this
    /// block's exact §4b nibble budget — the trace's "occasionally 930"
    /// padding case quantified.
    ///
    /// Per §4b the audio block size is "near-constant (927 bytes,
    /// occasionally 930)": the exact budget for the `comedian.amv` first
    /// block is `AMV_AUDIO_PREAMBLE_LEN + ceil(decoded_sample_count / 2)`
    /// = `8 + 919 = 927`, but a few blocks land at 930 — three bytes of
    /// trailing padding past the nibble-coded body. [`Self::is_consistent_with_body_len`]
    /// answers the boolean "is the on-disk length exactly the budget?";
    /// this companion turns that into a signed measurement so a recovery /
    /// inspection pass can tell *how* a block deviates rather than just
    /// *that* it does:
    ///
    /// * `Some(0)` — the exact §4b budget (the 927-byte common case).
    /// * `Some(n)`, `n > 0` — `n` trailing padding bytes past the nibble
    ///   body (the §4b "930" case reports `Some(3)`).
    /// * `None` — `total_payload_len` is shorter than the exact budget
    ///   (preamble + nibble body), i.e. the block is missing bytes its
    ///   declared `decoded_sample_count` requires — a short / truncated
    ///   block. A length below the 8-byte preamble itself also returns
    ///   `None` rather than panicking.
    ///
    /// `Some(0)` is exactly the case for which
    /// [`Self::is_consistent_with_body_len`] returns `true`; any other
    /// `Some(_)` is a padded block (still a valid, trace-recorded device
    /// shape) and `None` is a short block (invalid under §4b).
    ///
    /// Useful for a truncation-recovery / sanity pass that wants to
    /// distinguish trace-faithful padded blocks from genuinely clipped
    /// ones and to recover the padded slack without a second cast.
    pub fn body_padding_slack(&self, total_payload_len: u64) -> Option<u64> {
        let preamble = AMV_AUDIO_PREAMBLE_LEN as u64;
        let body_len = total_payload_len.checked_sub(preamble)?;
        body_len.checked_sub(self.nibble_body_len())
    }
}

/// Strict byte-shape validation of a `00dc` video-chunk payload against
/// the §4a invariants the trace records.
///
/// Per §4a every video chunk is a self-contained JPEG bracketed by
/// `FF D8` (SOI) at offset `0` and `FF D9` (EOI) at offset `size - 2`,
/// with no internal JPEG marker segments between them — the player
/// re-injects the stripped quant / Huffman tables before invoking its
/// hardcoded decoder, so the on-disk bitstream carries only the SOI /
/// EOI bracket plus entropy-coded data in between.
///
/// This helper checks the two byte-position invariants directly:
///
/// * `body[0..2] == FF D8` (SOI at chunk start).
/// * `body[size - 2..size] == FF D9` (EOI at chunk end).
///
/// Returns [`AmvDemuxerError::InvalidData`] when either invariant
/// fails, naming the offending byte position in the message. A
/// payload shorter than 4 bytes (no room for both markers) is also
/// rejected.
///
/// Useful for tooling that wants to confirm a recovered / extracted
/// video chunk has the §4a wire shape before handing it to a JPEG
/// decoder preprocessor. The demuxer's hot path does not invoke this
/// check (it forwards the raw bytes to downstream codec wiring per the
/// container's no-decode contract).
pub fn validate_video_payload_shape(body: &[u8]) -> Result<(), AmvDemuxerError> {
    if body.len() < 4 {
        return Err(AmvDemuxerError::InvalidData(format!(
            "00dc payload must be at least 4 bytes (SOI + EOI), got {}",
            body.len()
        )));
    }
    if body[0..2] != JPEG_SOI {
        return Err(AmvDemuxerError::InvalidData(format!(
            "00dc payload must begin with SOI (FF D8) at offset 0, got {:02X} {:02X}",
            body[0], body[1]
        )));
    }
    let end = body.len() - 2;
    if body[end..end + 2] != JPEG_EOI {
        return Err(AmvDemuxerError::InvalidData(format!(
            "00dc payload must end with EOI (FF D9) at offset {end}, got {:02X} {:02X}",
            body[end],
            body[end + 1]
        )));
    }
    Ok(())
}

/// Strict §4a invariant — confirm a `00dc` video chunk body carries
/// **no internal JPEG marker segments** between SOI and EOI.
///
/// The trace records that marker-scanning the first frame found *only
/// two* markers (SOI at +0 and EOI at `size − 2`): there is no
/// `APP0`/JFIF (`FF E0`), no `DQT` (`FF DB`), no `SOF0` (`FF C0`), no
/// `DHT` (`FF C4`) and no `SOS` (`FF DA`) marker segment. The
/// quantization / Huffman tables, frame geometry and scan parameters
/// are stripped and hardcoded in the player's decoder — the on-disk
/// bitstream is bare entropy-coded data wrapped in SOI..EOI.
///
/// This helper is the strict counterpart of
/// [`validate_video_payload_shape`]. After confirming the SOI / EOI
/// bracket via the shape check, it walks the entropy-coded payload
/// between offsets `2..(len − 2)` and reports the first byte position
/// at which an unexpected `FF xx` marker pair appears. The only `FF`
/// sequences allowed in the entropy section are:
///
/// * `FF 00` — JPEG byte stuffing (an `FF` data byte escaped by a
///   trailing `00`).
/// * `FF FF` — JPEG fill bytes (`0xFF` is the standard fill / pad
///   token between markers).
///
/// A trailing `FF` immediately preceding the closing `FF D9` EOI is
/// treated as a (non-stuffed) fill byte and accepted — the EOI itself
/// is checked by [`validate_video_payload_shape`] and excluded from
/// the entropy-scan window.
///
/// Returns [`AmvDemuxerError::InvalidData`] when an unexpected marker
/// is found, naming the marker byte (the second byte of the pair) and
/// the byte position relative to the start of the chunk body.
///
/// Strict callers (e.g. tooling that wants to confirm a recovered
/// video chunk really is the device profile's table-stripped variant)
/// should invoke [`validate_video_payload_shape`] first and then this
/// function; the demuxer's hot path invokes neither (it forwards raw
/// chunk bytes per the container's no-decode contract).
pub fn validate_video_payload_no_internal_markers(body: &[u8]) -> Result<(), AmvDemuxerError> {
    if body.len() < 4 {
        return Err(AmvDemuxerError::InvalidData(format!(
            "00dc payload must be at least 4 bytes (SOI + EOI), got {}",
            body.len()
        )));
    }
    // Entropy window: skip the 2-byte SOI at the start and the 2-byte
    // EOI at the end. The window may be empty for the degenerate
    // 4-byte payload, in which case there are no bytes to scan.
    let end_of_entropy = body.len() - 2;
    let mut i = 2;
    while i + 1 < end_of_entropy {
        if body[i] != 0xFF {
            i += 1;
            continue;
        }
        let next = body[i + 1];
        // `FF 00` is byte stuffing (escaped `FF` data byte); `FF FF`
        // is a fill byte that may legitimately appear between marker
        // pairs in standard JPEG streams. Both are permitted in the
        // entropy section.
        if next == 0x00 || next == 0xFF {
            i += 2;
            continue;
        }
        return Err(AmvDemuxerError::InvalidData(format!(
            "00dc payload must carry no internal JPEG markers; \
             found FF {next:02X} at offset {i}"
        )));
    }
    Ok(())
}

/// Strict §4 interleave invariant — confirm a sequence of observed
/// [`ChunkKind`]s walked out of the `movi` payload follows the trace's
/// recorded **strict 1:1 alternation, video-first** rule.
///
/// Per trace §4 every observed `.amv` file's `movi` payload is a flat
/// stream of `00dc` (video) / `01wb` (audio) leaf chunks under a rigid
/// pairing rule:
///
/// * Even-indexed chunks (`0`, `2`, `4`, …) carry `00dc` video frames.
/// * Odd-indexed chunks (`1`, `3`, `5`, …) carry `01wb` audio blocks.
/// * Each video frame is paired with exactly one audio block, so the
///   total chunk count is **even** — `comedian.amv` carries
///   `1116 + 1116 = 2232` chunks, `noel-son-lumiere.amv` carries
///   `2928 + 2928 = 5856` chunks.
///
/// This helper walks the supplied slice and returns:
///
/// * `Ok(())` when the sequence satisfies all three invariants: the
///   length is even, even positions are [`ChunkKind::Video`], odd
///   positions are [`ChunkKind::Audio`].
/// * `Err(AmvDemuxerError::InvalidData)` naming the first offending
///   chunk position and what was found there when the invariant fails.
///   A trailing unpaired video chunk (odd-length sequence ending with
///   `Video`) is reported as a missing audio chunk at the would-be
///   audio position.
///
/// An empty slice returns `Ok(())` — an empty `movi` payload has no
/// pairs to break.
///
/// `ChunkKind::Other(_)` is rejected at the offending position with
/// the tag bytes reported, since the trace records `00dc` and `01wb`
/// as the only tags observed inside `movi`.
///
/// Useful for tooling that wants to confirm a recovered / extracted
/// chunk sequence follows the device profile's rigid pairing rule —
/// for example, to flag a truncated file whose final video chunk has
/// no following audio block (the trace's worked example shows
/// `comedian.amv`'s walk ends 8 bytes before EOF on the trailer, with
/// the final chunk being an `01wb`, so a §4-conforming non-truncated
/// file must always end on an audio chunk). The demuxer's hot path
/// does not invoke this check (it forwards chunks as packets per the
/// container's no-decode contract); strictness is opt-in.
pub fn validate_movi_interleave(chunks: &[ChunkKind]) -> Result<(), AmvDemuxerError> {
    for (i, chunk) in chunks.iter().enumerate() {
        // Per §4: video on even positions, audio on odd positions.
        let expected_is_video = i % 2 == 0;
        match (chunk, expected_is_video) {
            (ChunkKind::Video, true) | (ChunkKind::Audio, false) => continue,
            (ChunkKind::Video, false) => {
                return Err(AmvDemuxerError::InvalidData(format!(
                    "movi chunk #{i} must be audio (01wb) per §4 strict \
                     1:1 alternation, got video (00dc)"
                )));
            }
            (ChunkKind::Audio, true) => {
                return Err(AmvDemuxerError::InvalidData(format!(
                    "movi chunk #{i} must be video (00dc) per §4 strict \
                     1:1 video-first alternation, got audio (01wb)"
                )));
            }
            (ChunkKind::Other(tag), _) => {
                return Err(AmvDemuxerError::InvalidData(format!(
                    "movi chunk #{i} must be 00dc or 01wb per §4 \
                     observed-tag set, got {:02X} {:02X} {:02X} {:02X}",
                    tag[0], tag[1], tag[2], tag[3]
                )));
            }
        }
    }
    // §4 strict 1:1 pairing rule: every video must have a paired audio
    // block, so the total count is even — an odd-length sequence ending
    // on a video chunk has a missing trailing audio block.
    if chunks.len() % 2 != 0 {
        return Err(AmvDemuxerError::InvalidData(format!(
            "movi chunk count must be even per §4 strict 1:1 pairing rule, \
             got {} chunks (missing trailing audio at #{})",
            chunks.len(),
            chunks.len()
        )));
    }
    Ok(())
}

// ────────────────────── typed movi payload walker ──────────────────────

/// Typed view of a single `movi` leaf chunk's payload, surfaced one at a
/// time by [`MoviPayloadIter`] as it walks an in-memory `movi`-body byte
/// buffer.
///
/// Each variant carries the chunk's file offset (relative to the start
/// of the `movi`-body buffer the iterator was constructed over) and the
/// chunk-payload byte slice borrowed from that buffer. The audio variant
/// additionally surfaces the already-parsed §4b 8-byte preamble so
/// strict callers don't have to re-parse it.
///
/// This is a **typed, opt-in** alternative to the demuxer's hot path
/// (which forwards raw bytes per the container's no-decode contract).
/// Tooling that wants to inspect every `00dc` JPEG payload or every
/// `01wb` ADPCM block at a typed surface — e.g. to run
/// [`validate_video_payload_no_internal_markers`] across the entire
/// stream, or to cross-check every preamble's `decoded_sample_count`
/// against the §4b frame-interval budget — calls [`MoviPayloadIter`]
/// instead of re-walking the raw bytes by hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoviPayload<'a> {
    /// `00dc` — video frame. Body is the bare JPEG `FF D8 … FF D9`
    /// payload (§4a: SOI..EOI bracket, no internal marker segments,
    /// tables hardcoded in the player). Bracket and no-internal-marker
    /// invariants are **not** enforced by the iterator — call
    /// [`validate_video_payload_shape`] /
    /// [`validate_video_payload_no_internal_markers`] on `body` if the
    /// caller wants strict §4a enforcement.
    Video {
        /// File offset of the chunk's 8-byte header, measured from the
        /// start of the `movi`-body buffer the iterator was constructed
        /// over. The chunk's payload bytes therefore start at
        /// `chunk_offset + 8` in that buffer.
        chunk_offset: usize,
        /// Borrowed body bytes, length matches the chunk's leaf-header
        /// `size` field (§4 no-padding rule).
        body: &'a [u8],
    },
    /// `01wb` — audio block. Body is the full §4b payload (8-byte
    /// preamble followed by the compressed nibble body). `preamble`
    /// is the already-parsed view of `body[0..8]`; `body` is the
    /// complete payload including the preamble so callers can compute
    /// the compressed-body length as `body.len() - AMV_AUDIO_PREAMBLE_LEN`
    /// without re-slicing.
    ///
    /// `AmvAudioPreamble::parse` is invoked by the iterator before
    /// emitting this variant, so a payload shorter than 8 bytes
    /// surfaces as an iteration error rather than a malformed
    /// `Audio` variant.
    Audio {
        /// File offset of the chunk's 8-byte header relative to the
        /// `movi`-body buffer the iterator was constructed over.
        chunk_offset: usize,
        /// Parsed §4b 8-byte preamble — `state` + `decoded_sample_count`.
        preamble: AmvAudioPreamble,
        /// Borrowed full payload bytes (preamble + compressed body).
        body: &'a [u8],
    },
    /// Any other 4-byte FOURCC tag. Real AMV files do not produce this
    /// in practice (the trace records only `00dc` and `01wb` inside
    /// `movi`), but the walker tolerates an unknown tag so the chunk
    /// surfaces as data rather than aborting iteration — strict
    /// callers should pair this iterator with
    /// [`validate_movi_interleave`] to reject the `Other` case.
    Other {
        /// File offset of the chunk's 8-byte header relative to the
        /// `movi`-body buffer the iterator was constructed over.
        chunk_offset: usize,
        /// Raw 4-byte FOURCC tag observed in the chunk header.
        tag: [u8; 4],
        /// Borrowed body bytes, length matches the chunk's leaf-header
        /// `size` field.
        body: &'a [u8],
    },
}

impl<'a> MoviPayload<'a> {
    /// Classify the payload as a [`ChunkKind`]. Mirrors
    /// [`ChunkHeader::kind`] but for the typed-payload view —
    /// convenient when feeding the iterator's output into
    /// [`validate_movi_interleave`].
    pub fn kind(&self) -> ChunkKind {
        match self {
            Self::Video { .. } => ChunkKind::Video,
            Self::Audio { .. } => ChunkKind::Audio,
            Self::Other { tag, .. } => ChunkKind::Other(*tag),
        }
    }

    /// File offset of the chunk's 8-byte leaf header, relative to the
    /// start of the `movi`-body buffer the iterator was constructed
    /// over. The chunk's payload bytes therefore start at
    /// `chunk_offset() + 8`.
    pub fn chunk_offset(&self) -> usize {
        match self {
            Self::Video { chunk_offset, .. }
            | Self::Audio { chunk_offset, .. }
            | Self::Other { chunk_offset, .. } => *chunk_offset,
        }
    }

    /// Borrowed payload bytes (the §4 leaf body, excluding the 8-byte
    /// header). For the `Audio` variant this includes the 8-byte
    /// preamble — callers that want only the compressed body should
    /// take `&body[AMV_AUDIO_PREAMBLE_LEN..]`.
    pub fn body(&self) -> &'a [u8] {
        match self {
            Self::Video { body, .. } | Self::Audio { body, .. } | Self::Other { body, .. } => body,
        }
    }
}

/// Walks an in-memory `movi`-body byte buffer producing typed
/// per-chunk payload views (§4 chunk framing + §4a/§4b per-chunk shape).
///
/// The input buffer must start at the byte **after** the `LIST <size>
/// 'movi'` FOURCC opener — i.e. the first byte of the first leaf chunk
/// header (the byte the demuxer's read cursor lands on after consuming
/// the 12-byte movi opener). The iterator advances by exactly `8 +
/// size` bytes per chunk under §4's no-padding rule and terminates on
/// one of three conditions:
///
/// * The cursor reaches end-of-buffer cleanly on an 8-byte boundary —
///   iteration ends with `None`.
/// * The next 8-byte window can't be read as a `ChunkHeader` (less
///   than 8 bytes remain) — iteration yields `Err(InvalidData)` once
///   then `None`.
/// * A chunk's declared `size` runs past the buffer — iteration yields
///   `Err(InvalidData)` once then `None`.
/// * An `01wb` payload is shorter than [`AMV_AUDIO_PREAMBLE_LEN`] —
///   iteration yields the preamble parse error once then `None`.
///
/// On an error the iterator latches a "done" flag so subsequent
/// `next()` calls return `None` rather than re-reporting the same
/// error — callers that want to surface the error should bind it on
/// first appearance.
///
/// The §4c `AMV_END_` ASCII trailer is **not** consumed by this
/// iterator — callers that want trailer-bounded walking should slice
/// the input buffer to exclude the trailing 8 ASCII bytes before
/// constructing the iterator (the demuxer's chunk-index path does
/// exactly this).
#[derive(Debug)]
pub struct MoviPayloadIter<'a> {
    /// The full `movi`-body byte buffer the iterator walks. Borrowed
    /// for the iterator's lifetime so emitted slices can borrow from
    /// it directly.
    buf: &'a [u8],
    /// Byte cursor into `buf`. Always advances by `8 + size` per
    /// emitted chunk under §4's no-padding rule.
    cursor: usize,
    /// Latched-done flag set after a successful end-of-buffer landing
    /// or after the first error — keeps `next()` returning `None`
    /// from then on.
    done: bool,
}

impl<'a> MoviPayloadIter<'a> {
    /// Construct an iterator over the supplied `movi`-body byte
    /// buffer. The buffer must start at the first leaf-chunk header
    /// (the byte after the 12-byte `LIST <size> 'movi'` opener) and
    /// must **not** include the §4c `AMV_END_` trailer.
    pub fn new(movi_body: &'a [u8]) -> Self {
        Self {
            buf: movi_body,
            cursor: 0,
            done: false,
        }
    }

    /// Byte cursor into the input buffer. Advances by `8 + size` after
    /// each emitted chunk. Useful for callers that want to report the
    /// offset at which iteration terminated.
    pub fn cursor(&self) -> usize {
        self.cursor
    }
}

impl<'a> Iterator for MoviPayloadIter<'a> {
    type Item = Result<MoviPayload<'a>, AmvDemuxerError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        // Clean end-of-buffer landing — every chunk consumed, no
        // trailing bytes left over.
        if self.cursor == self.buf.len() {
            self.done = true;
            return None;
        }
        // Less than a full 8-byte leaf header left — surface the
        // truncation as an InvalidData error so a malformed input
        // doesn't silently terminate iteration mid-stream.
        if self.cursor + 8 > self.buf.len() {
            self.done = true;
            return Some(Err(AmvDemuxerError::InvalidData(format!(
                "movi payload truncated at offset {}: {} bytes remain, \
                 need 8 for a chunk header",
                self.cursor,
                self.buf.len() - self.cursor,
            ))));
        }
        let header = match ChunkHeader::parse(&self.buf[self.cursor..self.cursor + 8]) {
            Ok(h) => h,
            Err(e) => {
                self.done = true;
                return Some(Err(e));
            }
        };
        let body_start = self.cursor + 8;
        let body_end = match body_start.checked_add(header.size as usize) {
            Some(e) => e,
            None => {
                self.done = true;
                return Some(Err(AmvDemuxerError::InvalidData(format!(
                    "movi chunk at offset {} declared size {} overflows usize",
                    self.cursor, header.size
                ))));
            }
        };
        if body_end > self.buf.len() {
            self.done = true;
            return Some(Err(AmvDemuxerError::InvalidData(format!(
                "movi chunk at offset {} declared size {} runs past buffer \
                 end (buffer is {} bytes, body would end at {})",
                self.cursor,
                header.size,
                self.buf.len(),
                body_end,
            ))));
        }
        let body = &self.buf[body_start..body_end];
        let chunk_offset = self.cursor;
        // Advance the cursor by `8 + size` per §4 (no even-byte padding).
        self.cursor = body_end;
        let payload = match header.kind() {
            ChunkKind::Video => MoviPayload::Video { chunk_offset, body },
            ChunkKind::Audio => match AmvAudioPreamble::parse(body) {
                Ok(preamble) => MoviPayload::Audio {
                    chunk_offset,
                    preamble,
                    body,
                },
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            },
            ChunkKind::Other(tag) => MoviPayload::Other {
                chunk_offset,
                tag,
                body,
            },
        };
        Some(Ok(payload))
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
    /// Parse the file prelude in strict mode — wraps [`Self::parse`] with
    /// the additional §2 + §3 sentinel checks documented in the trace:
    ///
    /// * Re-runs [`AmvHeader::validate_sentinels`] against the parsed
    ///   `amvh` body so a corrupted `dwMicroSecPerFrame` / `flag_one` /
    ///   `reserved_30` is rejected immediately rather than silently fed
    ///   downstream.
    /// * Verifies the three §3 stream-header bodies the trace records as
    ///   "entirely zero" really are entirely zero in the input slice —
    ///   the 56-byte video `strh` body, the 36-byte video `strf` body,
    ///   and the 48-byte audio `strh` body. Any non-zero byte in those
    ///   regions surfaces an [`AmvDemuxerError::InvalidData`] naming the
    ///   offending offset.
    /// * Re-runs [`AmvWaveFormat::validate_sentinels`] against the
    ///   parsed audio `strf` `WAVEFORMATEX` body so the six §3b
    ///   device-profile constants — `wFormatTag == 1`, `nChannels == 1`,
    ///   `nAvgBytesPerSec == nSamplesPerSec * 2`, `nBlockAlign == 2`,
    ///   `wBitsPerSample == 16`, `cbSize == 0` — are cross-checked.
    ///
    /// The permissive [`Self::parse`] path stays untouched so the
    /// existing demuxer-open path continues to accept any byte-shaped
    /// prelude that satisfies the §1-§4 FOURCC layout; this method is
    /// the opt-in cross-check for callers that want device-profile
    /// strictness.
    pub(crate) fn parse_strict(slice: &[u8]) -> Result<Self, AmvDemuxerError> {
        let prelude = Self::parse(slice)?;
        prelude.header.validate_sentinels()?;

        // §3a video strh body — 56 bytes immediately after the `strh`
        // FOURCC + size at file offset STRL_VIDEO_OFFSET + 12.
        let v_strh_body = STRL_VIDEO_OFFSET as usize + 20;
        require_all_zero(
            slice,
            v_strh_body,
            STRH_VIDEO_BODY_LEN as usize,
            "video strh body",
        )?;

        // §3a video strf body — 36 bytes (all-zero "no BITMAPINFOHEADER").
        let v_strf_body = v_strh_body + STRH_VIDEO_BODY_LEN as usize + 8;
        require_all_zero(
            slice,
            v_strf_body,
            STRF_VIDEO_BODY_LEN as usize,
            "video strf body",
        )?;

        // §3b audio strh body — 48 bytes (all-zero per §3b).
        let a_strh_body = v_strf_body + STRF_VIDEO_BODY_LEN as usize + 20;
        require_all_zero(
            slice,
            a_strh_body,
            STRH_AUDIO_BODY_LEN as usize,
            "audio strh body",
        )?;

        // §3b audio strf WAVEFORMATEX device-profile sentinels — six
        // fixed-value relationships the device hard-codes in the
        // 20-byte body (`wFormatTag`/`nChannels`/`nAvgBytesPerSec ==
        // nSamplesPerSec * 2` / `nBlockAlign` / `wBitsPerSample` /
        // `cbSize`). Run on the already-parsed view so this is a pure
        // semantic cross-check; the byte-shape check happened upstream.
        prelude.audio_format.validate_sentinels()?;

        Ok(prelude)
    }

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

/// Strict-mode helper: require an `len`-byte run starting at `at` to be
/// entirely zero. Used by [`AmvPrelude::parse_strict`] to verify that
/// the §3 all-zero stream-header bodies really are all zero in the
/// input slice. Reports the absolute offset of the first non-zero byte
/// for diagnostics.
fn require_all_zero(
    slice: &[u8],
    at: usize,
    len: usize,
    label: &str,
) -> Result<(), AmvDemuxerError> {
    if slice.len() < at + len {
        return Err(AmvDemuxerError::InvalidData(format!(
            "slice too short for {label} at offset {at:#x}"
        )));
    }
    for (i, &b) in slice[at..at + len].iter().enumerate() {
        if b != 0 {
            return Err(AmvDemuxerError::InvalidData(format!(
                "{label}: expected all-zero, found {b:#04x} at offset {:#x}",
                at + i
            )));
        }
    }
    Ok(())
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

    /// §2 trace doc table row: comedian device profile fixes
    /// micros_per_frame=83_333 (1e6/12), flag_one=1, reserved_30=0.
    /// `validate_sentinels` should accept this triple.
    #[test]
    fn validate_sentinels_accepts_comedian_profile() {
        let h = AmvHeader {
            micros_per_frame: 83_333,
            width: 128,
            height: 96,
            fps: 12,
            flag_one: 1,
            reserved_30: 0,
            duration_packed: 0x0000_0121,
        };
        h.validate_sentinels().expect("comedian sentinels accepted");
    }

    /// §2 trace doc cross-check row: noel device profile fixes
    /// micros_per_frame=62_500 (1e6/16), flag_one=1, reserved_30=0.
    #[test]
    fn validate_sentinels_accepts_noel_profile() {
        let h = AmvHeader {
            micros_per_frame: 62_500,
            width: 96,
            height: 64,
            fps: 16,
            flag_one: 1,
            reserved_30: 0,
            duration_packed: 0x0000_0302,
        };
        h.validate_sentinels().expect("noel sentinels accepted");
    }

    #[test]
    fn validate_sentinels_rejects_zero_fps() {
        let h = AmvHeader {
            micros_per_frame: 0,
            width: 128,
            height: 96,
            fps: 0,
            flag_one: 1,
            reserved_30: 0,
            duration_packed: 0,
        };
        let err = h.validate_sentinels().unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => assert!(msg.contains("fps")),
            other => panic!("expected InvalidData(fps), got {other:?}"),
        }
    }

    #[test]
    fn validate_sentinels_rejects_inconsistent_micros_per_frame() {
        // 12 fps with the noel-profile micros_per_frame (= 62_500) is
        // the canonical "header was tampered with" shape.
        let h = AmvHeader {
            micros_per_frame: 62_500,
            width: 128,
            height: 96,
            fps: 12,
            flag_one: 1,
            reserved_30: 0,
            duration_packed: 0x0000_0121,
        };
        let err = h.validate_sentinels().unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("dwMicroSecPerFrame"));
            }
            other => panic!("expected InvalidData(micros), got {other:?}"),
        }
    }

    #[test]
    fn validate_sentinels_rejects_wrong_flag_one() {
        let h = AmvHeader {
            micros_per_frame: 83_333,
            width: 128,
            height: 96,
            fps: 12,
            flag_one: 0,
            reserved_30: 0,
            duration_packed: 0x0000_0121,
        };
        let err = h.validate_sentinels().unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => assert!(msg.contains("+0x2C")),
            other => panic!("expected InvalidData(flag_one), got {other:?}"),
        }
    }

    #[test]
    fn validate_sentinels_rejects_non_zero_reserved_30() {
        let h = AmvHeader {
            micros_per_frame: 83_333,
            width: 128,
            height: 96,
            fps: 12,
            flag_one: 1,
            reserved_30: 0xDEAD_BEEF,
            duration_packed: 0x0000_0121,
        };
        let err = h.validate_sentinels().unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => assert!(msg.contains("+0x30")),
            other => panic!("expected InvalidData(reserved_30), got {other:?}"),
        }
    }

    #[test]
    fn prelude_parse_strict_accepts_comedian_synthetic() {
        let buf = build_synthetic_prelude(128, 96, 12, 0x0000_0121, 22_050);
        let p = AmvPrelude::parse_strict(&buf).expect("strict parse");
        assert_eq!(p.header.width, 128);
        assert_eq!(p.header.height, 96);
        assert_eq!(p.header.micros_per_frame, 83_333);
        assert_eq!(p.movi_payload_start, 0x13C);
    }

    /// Cross-fixture coverage: parse a synthetic prelude built for the
    /// noel device profile (96×64@16, micros_per_frame=62_500,
    /// packed_duration=0x0302 → 3:02). The synthetic-prelude builder
    /// derives `micros_per_frame` from `fps`, so this also exercises
    /// the §2 dwMicroSecPerFrame / fps cross-check inside
    /// `parse_strict`.
    #[test]
    fn prelude_parse_strict_accepts_noel_synthetic() {
        let buf = build_synthetic_prelude(96, 64, 16, 0x0000_0302, 22_050);
        let p = AmvPrelude::parse_strict(&buf).expect("strict parse");
        assert_eq!(p.header.width, 96);
        assert_eq!(p.header.height, 64);
        assert_eq!(p.header.fps, 16);
        assert_eq!(p.header.micros_per_frame, 62_500);
        assert_eq!(p.header.duration_packed, 0x0000_0302);
        // 3 * 60 + 2 = 182 seconds total.
        assert_eq!(p.header.duration_micros(), 182_000_000);
    }

    #[test]
    fn prelude_parse_strict_rejects_non_zero_video_strh_body() {
        let mut buf = build_synthetic_prelude(128, 96, 12, 0x0000_0121, 22_050);
        // The video strh body sits 20 bytes after STRL_VIDEO_OFFSET
        // (LIST 4 + size 4 + 'strl' 4 + 'strh' 4 + size 4).
        let v_strh_body = STRL_VIDEO_OFFSET as usize + 20;
        buf[v_strh_body] = 0x42;
        let err = AmvPrelude::parse_strict(&buf).unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("video strh body"));
            }
            other => panic!("expected InvalidData(strh), got {other:?}"),
        }
    }

    #[test]
    fn prelude_parse_strict_rejects_corrupted_flag_one() {
        let mut buf = build_synthetic_prelude(128, 96, 12, 0x0000_0121, 22_050);
        // amvh body starts at 0x20; flag_one is at body +0x2C → file 0x4C.
        buf[0x4C] = 0x02;
        let err = AmvPrelude::parse_strict(&buf).unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => assert!(msg.contains("+0x2C")),
            other => panic!("expected InvalidData(flag_one), got {other:?}"),
        }
    }

    /// §3b trace doc row: comedian / noel device profile fixes
    /// `wFormatTag=1`, `nChannels=1`, `nAvgBytesPerSec=nSamplesPerSec*2`,
    /// `nBlockAlign=2`, `wBitsPerSample=16`, `cbSize=0`.
    /// `AmvWaveFormat::validate_sentinels` should accept this tuple.
    #[test]
    fn waveformat_validate_sentinels_accepts_comedian_profile() {
        let fmt = AmvWaveFormat {
            format_tag: 1,
            channels: 1,
            samples_per_sec: 22_050,
            avg_bytes_per_sec: 44_100,
            block_align: 2,
            bits_per_sample: 16,
            cb_size: 0,
        };
        fmt.validate_sentinels()
            .expect("comedian audio strf sentinels accepted");
    }

    /// Cross-rate coverage — strict mode must still accept any
    /// `samples_per_sec` value as long as `avg_bytes_per_sec` is its
    /// `* 2` derivation. Use a hypothetical 44_100 Hz device profile to
    /// verify the rate itself is not gated.
    #[test]
    fn waveformat_validate_sentinels_accepts_arbitrary_samples_per_sec() {
        let fmt = AmvWaveFormat {
            format_tag: 1,
            channels: 1,
            samples_per_sec: 44_100,
            avg_bytes_per_sec: 88_200,
            block_align: 2,
            bits_per_sample: 16,
            cb_size: 0,
        };
        fmt.validate_sentinels()
            .expect("44k1 profile sentinels accepted");
    }

    #[test]
    fn waveformat_validate_sentinels_rejects_non_pcm_format_tag() {
        let fmt = AmvWaveFormat {
            format_tag: 2,
            channels: 1,
            samples_per_sec: 22_050,
            avg_bytes_per_sec: 44_100,
            block_align: 2,
            bits_per_sample: 16,
            cb_size: 0,
        };
        let err = fmt.validate_sentinels().unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("+0x00") && msg.contains("wFormatTag"));
            }
            other => panic!("expected InvalidData(wFormatTag), got {other:?}"),
        }
    }

    #[test]
    fn waveformat_validate_sentinels_rejects_stereo_channels() {
        let fmt = AmvWaveFormat {
            format_tag: 1,
            channels: 2,
            samples_per_sec: 22_050,
            avg_bytes_per_sec: 44_100,
            block_align: 2,
            bits_per_sample: 16,
            cb_size: 0,
        };
        let err = fmt.validate_sentinels().unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("+0x02") && msg.contains("nChannels"));
            }
            other => panic!("expected InvalidData(nChannels), got {other:?}"),
        }
    }

    #[test]
    fn waveformat_validate_sentinels_rejects_inconsistent_avg_bytes_per_sec() {
        // `samples_per_sec * 2` = 44 100; an attacker writing 22 050
        // (= samples_per_sec * 1, the on-disk nibble-rate misread) is
        // the canonical "header was tampered with" shape.
        let fmt = AmvWaveFormat {
            format_tag: 1,
            channels: 1,
            samples_per_sec: 22_050,
            avg_bytes_per_sec: 22_050,
            block_align: 2,
            bits_per_sample: 16,
            cb_size: 0,
        };
        let err = fmt.validate_sentinels().unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("+0x08") && msg.contains("nAvgBytesPerSec"));
            }
            other => panic!("expected InvalidData(nAvgBytesPerSec), got {other:?}"),
        }
    }

    #[test]
    fn waveformat_validate_sentinels_rejects_wrong_block_align() {
        let fmt = AmvWaveFormat {
            format_tag: 1,
            channels: 1,
            samples_per_sec: 22_050,
            avg_bytes_per_sec: 44_100,
            block_align: 4,
            bits_per_sample: 16,
            cb_size: 0,
        };
        let err = fmt.validate_sentinels().unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("+0x0C") && msg.contains("nBlockAlign"));
            }
            other => panic!("expected InvalidData(nBlockAlign), got {other:?}"),
        }
    }

    #[test]
    fn waveformat_validate_sentinels_rejects_wrong_bits_per_sample() {
        let fmt = AmvWaveFormat {
            format_tag: 1,
            channels: 1,
            samples_per_sec: 22_050,
            avg_bytes_per_sec: 44_100,
            block_align: 2,
            bits_per_sample: 8,
            cb_size: 0,
        };
        let err = fmt.validate_sentinels().unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("+0x0E") && msg.contains("wBitsPerSample"));
            }
            other => panic!("expected InvalidData(wBitsPerSample), got {other:?}"),
        }
    }

    #[test]
    fn waveformat_validate_sentinels_rejects_non_zero_cb_size() {
        let fmt = AmvWaveFormat {
            format_tag: 1,
            channels: 1,
            samples_per_sec: 22_050,
            avg_bytes_per_sec: 44_100,
            block_align: 2,
            bits_per_sample: 16,
            cb_size: 22,
        };
        let err = fmt.validate_sentinels().unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("+0x10") && msg.contains("cbSize"));
            }
            other => panic!("expected InvalidData(cbSize), got {other:?}"),
        }
    }

    /// Integration: strict prelude parse must reject when the audio
    /// strf WAVEFORMATEX violates a §3b sentinel — here, `wFormatTag`
    /// is flipped from 1 (PCM) to 0x55 (MP3, a value real AVI files
    /// can carry but no device-profile AMV emits).
    #[test]
    fn prelude_parse_strict_rejects_non_pcm_format_tag() {
        let mut buf = build_synthetic_prelude(128, 96, 12, 0x0000_0121, 22_050);
        // The audio strf body sits 20 bytes after the audio strl LIST.
        // Following the synthetic prelude layout exactly:
        //  STRL_VIDEO_OFFSET +  12               = video strh leaf
        //                    +  20 + 56 (strh)   = video strf leaf
        //                    +   8 + 36 (strf)   = end of video strl  (audio strl LIST starts)
        //                    +  12               = audio strh leaf
        //                    +  20 + 48 (strh)   = audio strf leaf
        //                    +   8               = audio strf body (= file offset of wFormatTag)
        let v = STRL_VIDEO_OFFSET as usize;
        let a_strl = v + 20 + STRH_VIDEO_BODY_LEN as usize + 8 + STRF_VIDEO_BODY_LEN as usize;
        let a_strf_body = a_strl + 20 + STRH_AUDIO_BODY_LEN as usize + 8;
        buf[a_strf_body] = 0x55;
        buf[a_strf_body + 1] = 0x00;
        let err = AmvPrelude::parse_strict(&buf).unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("wFormatTag"));
            }
            other => panic!("expected InvalidData(wFormatTag), got {other:?}"),
        }
    }

    /// Integration: strict prelude parse rejects when
    /// `nAvgBytesPerSec` is *not* `samples_per_sec * 2`. Build a
    /// 22_050-Hz prelude then patch `+0x08` to `samples_per_sec`
    /// directly (the on-disk nibble-byte-rate misread).
    #[test]
    fn prelude_parse_strict_rejects_inconsistent_avg_bytes_per_sec() {
        let mut buf = build_synthetic_prelude(128, 96, 12, 0x0000_0121, 22_050);
        let v = STRL_VIDEO_OFFSET as usize;
        let a_strl = v + 20 + STRH_VIDEO_BODY_LEN as usize + 8 + STRF_VIDEO_BODY_LEN as usize;
        let a_strf_body = a_strl + 20 + STRH_AUDIO_BODY_LEN as usize + 8;
        // Patch +0x08 from 44_100 to 22_050.
        buf[a_strf_body + 0x08..a_strf_body + 0x0C].copy_from_slice(&22_050u32.to_le_bytes());
        let err = AmvPrelude::parse_strict(&buf).unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("nAvgBytesPerSec"));
            }
            other => panic!("expected InvalidData(nAvgBytesPerSec), got {other:?}"),
        }
    }

    #[test]
    fn parse_strict_with_permissive_open_still_accepts_corrupted_input() {
        // The permissive `parse` path must NOT be tightened — verify
        // that a header which the strict path rejects (non-zero
        // reserved_30) still parses successfully through the
        // permissive entrypoint, so downstream consumers that opt out
        // of strictness keep their relaxed acceptance.
        let mut buf = build_synthetic_prelude(128, 96, 12, 0x0000_0121, 22_050);
        // reserved_30 lives at body +0x30 → file 0x50.
        buf[0x50] = 0x99;
        let permissive = AmvPrelude::parse(&buf).expect("permissive accepts");
        assert_eq!(permissive.header.reserved_30 & 0xFF, 0x99);
        let strict = AmvPrelude::parse_strict(&buf);
        assert!(strict.is_err());
    }

    /// §2 comedian profile: 0x21 = 33 s, 0x01 = 1 min, 0 hours →
    /// 0x0000_0121 round-trips through [`AmvDuration::to_packed`].
    #[test]
    fn duration_to_packed_round_trips_comedian() {
        let d = AmvDuration::from_packed(0x0000_0121);
        assert_eq!(d.to_packed(), 0x0000_0121);
    }

    /// §2 noel profile: 0x02 = 2 s, 0x03 = 3 min, 0 hours →
    /// 0x0000_0302 round-trips.
    #[test]
    fn duration_to_packed_round_trips_noel() {
        let d = AmvDuration::from_packed(0x0000_0302);
        assert_eq!(d.to_packed(), 0x0000_0302);
    }

    /// `to_packed` always writes the +0x37 reserved byte as 0 per §2.
    #[test]
    fn duration_to_packed_writes_reserved_byte_zero() {
        let d = AmvDuration {
            seconds: 1,
            minutes: 2,
            hours: 3,
        };
        let bytes = d.to_packed().to_le_bytes();
        assert_eq!(bytes[3], 0);
    }

    /// total_seconds matches the comedian §2 worked example (1:33 = 93 s).
    #[test]
    fn duration_total_seconds_matches_comedian() {
        let d = AmvDuration::from_packed(0x0000_0121);
        assert_eq!(d.total_seconds(), 60 + 33);
    }

    /// total_seconds matches the noel §2 worked example (3:02 = 182 s).
    #[test]
    fn duration_total_seconds_matches_noel() {
        let d = AmvDuration::from_packed(0x0000_0302);
        assert_eq!(d.total_seconds(), 3 * 60 + 2);
    }

    // ── §2 frame-count derivation helpers ─────────────────────────

    /// §2 worked example: comedian.amv has 1116 video chunks at 12 fps
    /// → 1116 / 12 = 93 s = 1:33 → `0x0000_0121`. `from_frame_count`
    /// must reproduce the packed bytes the device wrote at `amvh
    /// +0x34` from the chunk count alone.
    #[test]
    fn duration_from_frame_count_matches_comedian_worked_example() {
        let d = AmvDuration::from_frame_count(1116, 12);
        assert_eq!(d.seconds, 0x21);
        assert_eq!(d.minutes, 0x01);
        assert_eq!(d.hours, 0x00);
        assert_eq!(d.to_packed(), 0x0000_0121);
    }

    /// §2 worked example: noel-son-lumiere.amv has 2928 video chunks at
    /// 16 fps → 2928 / 16 = 183 s = 3:03. The trace records its packed
    /// duration as `0x0000_0302` (3:02) because the original 3:02
    /// source clip was transcoded into 2928 video chunks at 16 fps,
    /// which lands the integer-divided duration one second past the
    /// source's stated 3:02. This test pins the *derivation*
    /// (`from_frame_count`) rather than the parsed header value so the
    /// helper's arithmetic is locked to the trace's documented
    /// behaviour — that the muxer applies whole-second flooring of
    /// `frame_count / fps`. The cross-check test below confirms the
    /// `is_consistent_with_frame_count` helper rejects the off-by-one.
    #[test]
    fn duration_from_frame_count_floors_whole_seconds_for_noel_chunks() {
        let d = AmvDuration::from_frame_count(2928, 16);
        // 2928 / 16 = 183 s = 3 min 3 s
        assert_eq!(d.seconds, 3);
        assert_eq!(d.minutes, 3);
        assert_eq!(d.hours, 0);
    }

    /// `from_frame_count` rolls past one hour into the `hours`
    /// component. 60 min × 12 fps × 60 s/min = 43200 frames at 12 fps
    /// = 3600 s = 1 h 0 m 0 s.
    #[test]
    fn duration_from_frame_count_rolls_into_hours_component() {
        let d = AmvDuration::from_frame_count(43_200, 12);
        assert_eq!(d.seconds, 0);
        assert_eq!(d.minutes, 0);
        assert_eq!(d.hours, 1);
    }

    /// §2 zero-fps guard: `from_frame_count` returns the all-zero
    /// duration when `fps == 0` instead of dividing by zero. Higher
    /// layers (`AmvHeader::validate_sentinels`) reject `fps == 0`
    /// outright; this guard exists only so the helper itself stays
    /// infallible.
    #[test]
    fn duration_from_frame_count_returns_zero_for_zero_fps() {
        let d = AmvDuration::from_frame_count(1116, 0);
        assert_eq!(d.seconds, 0);
        assert_eq!(d.minutes, 0);
        assert_eq!(d.hours, 0);
    }

    /// `from_frame_count` saturates at `255 h` when the computed hours
    /// component overflows a `u8`. 256 h × 3600 s/h × 1 fps = 921 600
    /// frames; the resulting duration clamps `hours` to 255.
    #[test]
    fn duration_from_frame_count_saturates_hours_component() {
        let d = AmvDuration::from_frame_count(256 * 3600, 1);
        assert_eq!(d.hours, u8::MAX);
    }

    /// `is_consistent_with_frame_count` accepts the §2 comedian
    /// worked-example pair (1116 frames, 12 fps) when the duration was
    /// parsed from the device-written `0x0000_0121` packed bytes.
    #[test]
    fn duration_is_consistent_accepts_comedian_pair() {
        let d = AmvDuration::from_packed(0x0000_0121);
        assert!(d.is_consistent_with_frame_count(1116, 12));
    }

    /// `is_consistent_with_frame_count` rejects the off-by-one mismatch
    /// the noel-son-lumiere.amv profile exhibits: the parsed header
    /// reads `0x0000_0302` (3:02, matching the source clip's stated
    /// duration) but the file carries 2928 chunks at 16 fps which
    /// integer-divides to 3:03. The helper's job is to surface exactly
    /// this kind of header-vs-chunk-count discrepancy.
    #[test]
    fn duration_is_consistent_rejects_noel_header_vs_chunk_count_mismatch() {
        let parsed = AmvDuration::from_packed(0x0000_0302);
        assert!(!parsed.is_consistent_with_frame_count(2928, 16));
    }

    /// `is_consistent_with_frame_count` rejects the comedian profile
    /// when the supplied frame count is one short of the recorded
    /// 1116 — the truncation case the helper is designed to catch.
    #[test]
    fn duration_is_consistent_rejects_one_short_of_comedian() {
        let parsed = AmvDuration::from_packed(0x0000_0121);
        // 1115 / 12 = 92 s → seconds=32 min=1, mismatching parsed (33,1).
        assert!(!parsed.is_consistent_with_frame_count(1115, 12));
    }

    /// `is_consistent_with_frame_count` returns `false` when `fps == 0`
    /// unless the parsed duration is also the all-zero triple,
    /// mirroring `from_frame_count`'s zero-fps guard.
    #[test]
    fn duration_is_consistent_zero_fps_only_matches_zero_duration() {
        let zero = AmvDuration::from_packed(0);
        assert!(zero.is_consistent_with_frame_count(0, 0));
        assert!(zero.is_consistent_with_frame_count(1116, 0));
        let nonzero = AmvDuration::from_packed(0x0000_0121);
        assert!(!nonzero.is_consistent_with_frame_count(1116, 0));
    }

    /// `from_frame_count` followed by `to_packed` reproduces the
    /// little-endian dword the device wrote for both fixtures' frame
    /// counts (comedian 1116 / 12 fps → `0x0000_0121`; noel 2928 / 16
    /// fps → `0x0000_0303` — note the device's stored `0x0302` is the
    /// trace-recorded source-clip duration, not the integer-divided
    /// chunk-count derivation, which is exactly the discrepancy
    /// `is_consistent_with_frame_count` is designed to surface).
    #[test]
    fn duration_from_frame_count_then_to_packed_round_trips() {
        assert_eq!(
            AmvDuration::from_frame_count(1116, 12).to_packed(),
            0x0000_0121
        );
        assert_eq!(
            AmvDuration::from_frame_count(2928, 16).to_packed(),
            0x0000_0303
        );
    }

    // ── §4a video payload shape validator ──────────────────────────

    /// §4a: the first video chunk of the comedian fixture (1633 bytes)
    /// starts with `FF D8` and ends with `FF D9`. Build the same shape
    /// synthetically and confirm the validator accepts it.
    #[test]
    fn video_payload_shape_accepts_soi_eoi_bracket() {
        let mut body = vec![0u8; 1633];
        body[0..2].copy_from_slice(&JPEG_SOI);
        body[1631..1633].copy_from_slice(&JPEG_EOI);
        validate_video_payload_shape(&body).expect("SOI..EOI bracket accepted");
    }

    /// §4a minimum payload: 4 bytes = SOI + EOI, no body in between
    /// (degenerate but byte-shape-valid).
    #[test]
    fn video_payload_shape_accepts_minimum_4_byte_payload() {
        let body = [0xFF, 0xD8, 0xFF, 0xD9];
        validate_video_payload_shape(&body).expect("4-byte SOI+EOI accepted");
    }

    /// Payload shorter than 4 bytes cannot carry both markers — reject.
    #[test]
    fn video_payload_shape_rejects_three_byte_payload() {
        let body = [0xFF, 0xD8, 0xFF];
        let err = validate_video_payload_shape(&body).unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => assert!(msg.contains("at least 4 bytes")),
            other => panic!("expected InvalidData(short), got {other:?}"),
        }
    }

    /// Wrong start marker (corrupted first byte) rejected with offset 0.
    #[test]
    fn video_payload_shape_rejects_wrong_soi() {
        let mut body = vec![0xAB, 0xCD];
        body.extend_from_slice(&JPEG_EOI);
        let err = validate_video_payload_shape(&body).unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("SOI"));
                assert!(msg.contains("offset 0"));
            }
            other => panic!("expected InvalidData(SOI), got {other:?}"),
        }
    }

    /// Wrong end marker rejected, with the end offset reported.
    #[test]
    fn video_payload_shape_rejects_wrong_eoi() {
        let mut body = vec![0u8; 16];
        body[0..2].copy_from_slice(&JPEG_SOI);
        // Leave the last two bytes as 00 00 — should reject.
        let err = validate_video_payload_shape(&body).unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("EOI"));
                assert!(msg.contains("14"));
            }
            other => panic!("expected InvalidData(EOI), got {other:?}"),
        }
    }

    // ── §4a strict-marker validator ─────────────────────────────────

    /// §4a: a bare SOI..EOI bracket with no entropy payload is the
    /// degenerate empty case — the entropy window is zero bytes wide,
    /// so the no-internal-marker walker has nothing to scan and
    /// accepts trivially.
    #[test]
    fn no_internal_markers_accepts_empty_entropy_window() {
        let body = [0xFF, 0xD8, 0xFF, 0xD9];
        validate_video_payload_no_internal_markers(&body)
            .expect("4-byte SOI+EOI accepted (no entropy)");
    }

    /// §4a: the trace's "immediately after SOI the bytes are
    /// entropy-coded data" — synthesise a SOI..entropy..EOI body
    /// whose entropy region contains only non-`FF` data bytes. The
    /// walker accepts it.
    #[test]
    fn no_internal_markers_accepts_plain_entropy_bytes() {
        // From the trace's frame-0 first-bytes example
        // (`FF D8 E6 49 A6 93 …`): SOI followed by non-FF entropy.
        let mut body = vec![0xFF, 0xD8, 0xE6, 0x49, 0xA6, 0x93, 0x12, 0x34];
        body.extend_from_slice(&JPEG_EOI);
        validate_video_payload_no_internal_markers(&body).expect("non-FF entropy bytes accepted");
    }

    /// §4a: `FF 00` byte stuffing is permitted — an entropy byte that
    /// happens to be `0xFF` is escaped by a trailing `0x00`. The
    /// scanner must skip the pair without flagging it as a marker.
    #[test]
    fn no_internal_markers_accepts_byte_stuffed_ff_00() {
        let mut body = vec![0xFF, 0xD8, 0xAB, 0xFF, 0x00, 0xCD];
        body.extend_from_slice(&JPEG_EOI);
        validate_video_payload_no_internal_markers(&body).expect("FF 00 byte stuffing accepted");
    }

    /// §4a: a `FF FF` fill-byte pair is permitted in the entropy
    /// window. JPEG allows `0xFF` to repeat as a fill / pad token,
    /// and the trace does not explicitly forbid it; the scanner
    /// must skip past the pair without flagging it as a marker.
    #[test]
    fn no_internal_markers_accepts_ff_ff_fill() {
        let mut body = vec![0xFF, 0xD8, 0xAB, 0xFF, 0xFF, 0xCD];
        body.extend_from_slice(&JPEG_EOI);
        validate_video_payload_no_internal_markers(&body).expect("FF FF fill bytes accepted");
    }

    /// §4a: `FF E0` (APP0/JFIF) inside the entropy window MUST be
    /// rejected — the trace lists APP0 explicitly as a marker that
    /// the device-stripped bitstream does not carry.
    #[test]
    fn no_internal_markers_rejects_app0() {
        let mut body = vec![0xFF, 0xD8, 0xAB, 0xFF, 0xE0, 0xCD];
        body.extend_from_slice(&JPEG_EOI);
        let err = validate_video_payload_no_internal_markers(&body).unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("FF E0"), "expected APP0 marker named: {msg}");
                assert!(
                    msg.contains("offset 3"),
                    "expected offset 3 reported: {msg}"
                );
            }
            other => panic!("expected InvalidData(APP0), got {other:?}"),
        }
    }

    /// §4a: `FF DB` (DQT — quant-table segment) inside the entropy
    /// window MUST be rejected. The trace lists DQT explicitly as
    /// a marker absent from the device-stripped bitstream.
    #[test]
    fn no_internal_markers_rejects_dqt() {
        let mut body = vec![0xFF, 0xD8, 0x00, 0xFF, 0xDB, 0x00];
        body.extend_from_slice(&JPEG_EOI);
        let err = validate_video_payload_no_internal_markers(&body).unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("FF DB"), "expected DQT marker named: {msg}");
            }
            other => panic!("expected InvalidData(DQT), got {other:?}"),
        }
    }

    /// §4a: `FF C0` (SOF0 — frame-geometry segment) inside the
    /// entropy window MUST be rejected.
    #[test]
    fn no_internal_markers_rejects_sof0() {
        let mut body = vec![0xFF, 0xD8, 0x00, 0xFF, 0xC0, 0x00];
        body.extend_from_slice(&JPEG_EOI);
        let err = validate_video_payload_no_internal_markers(&body).unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("FF C0"), "expected SOF0 marker named: {msg}");
            }
            other => panic!("expected InvalidData(SOF0), got {other:?}"),
        }
    }

    /// §4a: `FF C4` (DHT — Huffman-table segment) inside the entropy
    /// window MUST be rejected.
    #[test]
    fn no_internal_markers_rejects_dht() {
        let mut body = vec![0xFF, 0xD8, 0x00, 0xFF, 0xC4, 0x00];
        body.extend_from_slice(&JPEG_EOI);
        let err = validate_video_payload_no_internal_markers(&body).unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("FF C4"), "expected DHT marker named: {msg}");
            }
            other => panic!("expected InvalidData(DHT), got {other:?}"),
        }
    }

    /// §4a: `FF DA` (SOS — start-of-scan segment) inside the entropy
    /// window MUST be rejected.
    #[test]
    fn no_internal_markers_rejects_sos() {
        let mut body = vec![0xFF, 0xD8, 0x00, 0xFF, 0xDA, 0x00];
        body.extend_from_slice(&JPEG_EOI);
        let err = validate_video_payload_no_internal_markers(&body).unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("FF DA"), "expected SOS marker named: {msg}");
            }
            other => panic!("expected InvalidData(SOS), got {other:?}"),
        }
    }

    /// §4a: an EOI-style `FF D9` inside the entropy window (i.e. NOT
    /// the closing bracket) is itself an unexpected marker — the
    /// trace records exactly one EOI per frame, at `size − 2`.
    #[test]
    fn no_internal_markers_rejects_premature_eoi() {
        let mut body = vec![0xFF, 0xD8, 0xAB, 0xFF, 0xD9, 0xCD];
        body.extend_from_slice(&JPEG_EOI);
        let err = validate_video_payload_no_internal_markers(&body).unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("FF D9"), "expected premature-EOI named: {msg}");
                assert!(
                    msg.contains("offset 3"),
                    "expected offset 3 reported: {msg}"
                );
            }
            other => panic!("expected InvalidData(premature EOI), got {other:?}"),
        }
    }

    /// §4a: a payload shorter than the 4-byte minimum (SOI + EOI) is
    /// rejected by the strict scanner with the same length error
    /// shape as the shape validator.
    #[test]
    fn no_internal_markers_rejects_short_payload() {
        let body = [0xFF, 0xD8, 0xFF];
        let err = validate_video_payload_no_internal_markers(&body).unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => assert!(msg.contains("at least 4 bytes")),
            other => panic!("expected InvalidData(short), got {other:?}"),
        }
    }

    // ── §4b audio preamble parser + validator ──────────────────────

    /// §4b comedian first-block observation: state=0, samples=1837.
    #[test]
    fn audio_preamble_parse_comedian_first_block() {
        let mut body = vec![0u8; AMV_AUDIO_PREAMBLE_LEN];
        body[0..4].copy_from_slice(&0u32.to_le_bytes());
        body[4..8].copy_from_slice(&1837u32.to_le_bytes());
        let p = AmvAudioPreamble::parse(&body).unwrap();
        assert_eq!(p.state, 0);
        assert_eq!(p.decoded_sample_count, 1837);
    }

    /// §4b non-zero state — the trace records the state field varies
    /// block-to-block; the parser surfaces the value verbatim.
    #[test]
    fn audio_preamble_parse_surfaces_nonzero_state() {
        let mut body = vec![0u8; AMV_AUDIO_PREAMBLE_LEN];
        body[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        body[4..8].copy_from_slice(&1837u32.to_le_bytes());
        let p = AmvAudioPreamble::parse(&body).unwrap();
        assert_eq!(p.state, 0xDEAD_BEEF);
        assert_eq!(p.decoded_sample_count, 1837);
    }

    /// Slice shorter than 8 bytes cannot carry the preamble.
    #[test]
    fn audio_preamble_parse_rejects_short_slice() {
        let body = vec![0u8; 7];
        let err = AmvAudioPreamble::parse(&body).unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => assert!(msg.contains("8 bytes")),
            other => panic!("expected InvalidData(short), got {other:?}"),
        }
    }

    /// `validate_sentinels` accepts the comedian first-block shape.
    #[test]
    fn audio_preamble_validate_accepts_comedian_first_block() {
        let p = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 1837,
        };
        p.validate_sentinels()
            .expect("comedian first-block preamble accepted");
    }

    /// `validate_sentinels` accepts the noel first-block shape
    /// (22 050 ÷ 16 ≈ 1378).
    #[test]
    fn audio_preamble_validate_accepts_noel_first_block() {
        let p = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 1378,
        };
        p.validate_sentinels()
            .expect("noel first-block preamble accepted");
    }

    /// `validate_sentinels` rejects a zero decoded sample count — no
    /// observed device profile emits an empty audio block.
    #[test]
    fn audio_preamble_validate_rejects_zero_sample_count() {
        let p = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 0,
        };
        let err = p.validate_sentinels().unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(msg.contains("+0x04"));
                assert!(msg.contains("decoded_sample_count"));
            }
            other => panic!("expected InvalidData(samples), got {other:?}"),
        }
    }

    /// `validate_sentinels` does NOT gate on the state field (which the
    /// trace records as per-block-varying). A non-zero state passes.
    #[test]
    fn audio_preamble_validate_accepts_arbitrary_state() {
        let p = AmvAudioPreamble {
            state: 0xCAFEBABE,
            decoded_sample_count: 1837,
        };
        p.validate_sentinels()
            .expect("non-zero state must not gate sentinel validation");
    }

    // ── §4b ↔ §3b ↔ §2 frame-interval cross-check helpers ──────────

    /// `AmvWaveFormat::frame_interval_samples` reproduces the trace's
    /// §4b worked example exactly: 22 050 Hz ÷ 12 fps = 1837 — the
    /// comedian first audio block's `decoded_sample_count`.
    #[test]
    fn waveformat_frame_interval_samples_matches_comedian_worked_example() {
        let fmt = AmvWaveFormat {
            format_tag: 1,
            channels: 1,
            samples_per_sec: 22_050,
            avg_bytes_per_sec: 44_100,
            block_align: 2,
            bits_per_sample: 16,
            cb_size: 0,
        };
        assert_eq!(fmt.frame_interval_samples(12), 1837);
    }

    /// Cross-rate coverage — 22 050 Hz ÷ 16 fps = 1378 (the noel
    /// profile's expected per-block budget under the §4 1:1 interleave).
    #[test]
    fn waveformat_frame_interval_samples_matches_noel_worked_example() {
        let fmt = AmvWaveFormat {
            format_tag: 1,
            channels: 1,
            samples_per_sec: 22_050,
            avg_bytes_per_sec: 44_100,
            block_align: 2,
            bits_per_sample: 16,
            cb_size: 0,
        };
        assert_eq!(fmt.frame_interval_samples(16), 1378);
    }

    /// `frame_interval_samples` short-circuits to `0` on `fps == 0`
    /// rather than dividing by zero.
    #[test]
    fn waveformat_frame_interval_samples_returns_zero_when_fps_is_zero() {
        let fmt = AmvWaveFormat {
            format_tag: 1,
            channels: 1,
            samples_per_sec: 22_050,
            avg_bytes_per_sec: 44_100,
            block_align: 2,
            bits_per_sample: 16,
            cb_size: 0,
        };
        assert_eq!(fmt.frame_interval_samples(0), 0);
    }

    /// `AmvAudioPreamble::is_consistent_with_frame_interval` returns
    /// `true` on the trace's §4b worked example: comedian's first block
    /// holds `decoded_sample_count = 1837`, matching `22 050 ÷ 12`.
    #[test]
    fn audio_preamble_consistent_with_comedian_frame_interval() {
        let p = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 1837,
        };
        assert!(p.is_consistent_with_frame_interval(22_050, 12));
    }

    /// Cross-rate coverage — the noel profile's expected first-block
    /// budget is `22 050 ÷ 16 = 1378`.
    #[test]
    fn audio_preamble_consistent_with_noel_frame_interval() {
        let p = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 1378,
        };
        assert!(p.is_consistent_with_frame_interval(22_050, 16));
    }

    /// A preamble whose `decoded_sample_count` mismatches the frame
    /// interval budget (here, the noel block claiming the comedian
    /// budget) reports inconsistent.
    #[test]
    fn audio_preamble_consistent_rejects_mismatched_sample_count() {
        let p = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 1837,
        };
        assert!(!p.is_consistent_with_frame_interval(22_050, 16));
    }

    /// `fps == 0` is gated to `false` unless the sample count is also
    /// `0` (matching the `frame_interval_samples` zero-fps short-circuit).
    #[test]
    fn audio_preamble_consistent_handles_zero_fps() {
        let p_nonzero = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 1837,
        };
        assert!(!p_nonzero.is_consistent_with_frame_interval(22_050, 0));

        let p_zero = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 0,
        };
        assert!(p_zero.is_consistent_with_frame_interval(22_050, 0));
    }

    /// End-to-end derivation cross-check: a preamble whose
    /// `decoded_sample_count` matches `format.frame_interval_samples(fps)`
    /// must satisfy `is_consistent_with_frame_interval(format.samples_per_sec,
    /// fps)`. This pins the two helpers' arithmetic against each other.
    #[test]
    fn audio_preamble_consistent_matches_waveformat_helper() {
        let fmt = AmvWaveFormat {
            format_tag: 1,
            channels: 1,
            samples_per_sec: 22_050,
            avg_bytes_per_sec: 44_100,
            block_align: 2,
            bits_per_sample: 16,
            cb_size: 0,
        };
        for fps in [12, 16, 24, 30] {
            let p = AmvAudioPreamble {
                state: 0,
                decoded_sample_count: fmt.frame_interval_samples(fps),
            };
            assert!(
                p.is_consistent_with_frame_interval(fmt.samples_per_sec, fps),
                "fps={fps} must round-trip via frame_interval_samples"
            );
        }
    }

    /// §4b nibble budget — the comedian first block's `1837` decoded
    /// samples pack into `ceil(1837 / 2) = 919` compressed body bytes,
    /// exactly the trace's recorded body length.
    #[test]
    fn audio_preamble_nibble_body_len_comedian_first_block() {
        let p = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 1837,
        };
        assert_eq!(p.nibble_body_len(), 919);
    }

    /// §4b nibble budget — an even sample count packs into exactly
    /// `n / 2` body bytes with no ceiling round-up.
    #[test]
    fn audio_preamble_nibble_body_len_even_sample_count() {
        let p = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 1836,
        };
        assert_eq!(p.nibble_body_len(), 918);
    }

    /// §4b nibble budget — a lone trailing sample still occupies a full
    /// byte (one used nibble + one pad nibble), so `1` sample needs `1`
    /// body byte.
    #[test]
    fn audio_preamble_nibble_body_len_rounds_up_odd() {
        let p = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 1,
        };
        assert_eq!(p.nibble_body_len(), 1);
    }

    /// §4b nibble budget — a zero sample count needs zero body bytes.
    #[test]
    fn audio_preamble_nibble_body_len_zero() {
        let p = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 0,
        };
        assert_eq!(p.nibble_body_len(), 0);
    }

    /// §4b block-length cross-check — the comedian first block's full
    /// `01wb` payload of `927` bytes (`8` preamble + `919` body) is
    /// consistent with its declared `1837` decoded samples.
    #[test]
    fn audio_preamble_consistent_body_len_comedian() {
        let p = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 1837,
        };
        assert!(p.is_consistent_with_body_len(927));
    }

    /// §4b block-length cross-check — a payload one byte short of the
    /// nibble budget (a body clipped mid-write) is rejected.
    #[test]
    fn audio_preamble_consistent_body_len_rejects_short_body() {
        let p = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 1837,
        };
        assert!(!p.is_consistent_with_body_len(926));
    }

    /// §4b block-length cross-check — a payload one byte over the nibble
    /// budget is also rejected (the relation is exact, not a lower bound).
    #[test]
    fn audio_preamble_consistent_body_len_rejects_long_body() {
        let p = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 1837,
        };
        assert!(!p.is_consistent_with_body_len(928));
    }

    /// §4b block-length cross-check — a `total_payload_len` shorter than
    /// the 8-byte preamble cannot carry a valid block and returns `false`
    /// rather than panicking on the underflow.
    #[test]
    fn audio_preamble_consistent_body_len_rejects_sub_preamble() {
        let p = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 1837,
        };
        assert!(!p.is_consistent_with_body_len(7));
        // The boundary case: exactly the preamble with a zero-length body
        // only matches a zero-sample block.
        assert!(!p.is_consistent_with_body_len(8));
        let empty = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 0,
        };
        assert!(empty.is_consistent_with_body_len(8));
    }

    /// §4b end-to-end — `is_consistent_with_body_len` is the preamble
    /// length plus `nibble_body_len`, pinned against each other for the
    /// odd-sample-count round-up case.
    #[test]
    fn audio_preamble_consistent_body_len_matches_nibble_helper() {
        for samples in [0u32, 1, 2, 3, 1836, 1837, 1378] {
            let p = AmvAudioPreamble {
                state: 0,
                decoded_sample_count: samples,
            };
            let expected_total = AMV_AUDIO_PREAMBLE_LEN as u64 + p.nibble_body_len();
            assert!(
                p.is_consistent_with_body_len(expected_total),
                "samples={samples} expected_total={expected_total}"
            );
            assert!(!p.is_consistent_with_body_len(expected_total + 1));
        }
    }

    /// §4b padding slack — the comedian first block at the exact `927`
    /// budget reports zero slack.
    #[test]
    fn audio_preamble_padding_slack_exact_budget() {
        let p = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 1837,
        };
        assert_eq!(p.body_padding_slack(927), Some(0));
    }

    /// §4b padding slack — the trace's "occasionally 930" padded block
    /// reports three bytes of trailing padding past the `927` nibble
    /// budget.
    #[test]
    fn audio_preamble_padding_slack_padded_930() {
        let p = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 1837,
        };
        assert_eq!(p.body_padding_slack(930), Some(3));
    }

    /// §4b padding slack — a block one byte short of its nibble budget
    /// (clipped mid-write) reports `None`, not a wrapped underflow.
    #[test]
    fn audio_preamble_padding_slack_short_body_is_none() {
        let p = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 1837,
        };
        assert_eq!(p.body_padding_slack(926), None);
    }

    /// §4b padding slack — a `total_payload_len` below the 8-byte
    /// preamble returns `None` rather than panicking on the underflow.
    #[test]
    fn audio_preamble_padding_slack_sub_preamble_is_none() {
        let p = AmvAudioPreamble {
            state: 0,
            decoded_sample_count: 1837,
        };
        assert_eq!(p.body_padding_slack(7), None);
        assert_eq!(p.body_padding_slack(0), None);
    }

    /// §4b padding slack — `Some(0)` is exactly the set of lengths for
    /// which `is_consistent_with_body_len` returns `true`, and any other
    /// `Some(_)` is a padded (still valid) block, across the odd / even /
    /// zero sample-count cases.
    #[test]
    fn audio_preamble_padding_slack_agrees_with_consistency() {
        for samples in [0u32, 1, 2, 3, 1836, 1837, 1378] {
            let p = AmvAudioPreamble {
                state: 0,
                decoded_sample_count: samples,
            };
            let exact = AMV_AUDIO_PREAMBLE_LEN as u64 + p.nibble_body_len();
            assert_eq!(p.body_padding_slack(exact), Some(0), "samples={samples}");
            assert!(p.is_consistent_with_body_len(exact));
            // A padded block: slack is the number of bytes past the budget,
            // and the exact-relation boolean correctly reports false.
            assert_eq!(
                p.body_padding_slack(exact + 3),
                Some(3),
                "samples={samples}"
            );
            assert!(!p.is_consistent_with_body_len(exact + 3));
            // A short block: None, and the boolean is false.
            if exact > 0 {
                assert_eq!(p.body_padding_slack(exact - 1), None, "samples={samples}");
                assert!(!p.is_consistent_with_body_len(exact - 1));
            }
        }
    }

    /// §4 strict 1:1 video-first alternation — empty slice accepted
    /// (no pairs to break).
    #[test]
    fn validate_movi_interleave_accepts_empty() {
        validate_movi_interleave(&[]).expect("empty movi must pass");
    }

    /// §4: smallest valid pair (one video, one audio) is accepted.
    #[test]
    fn validate_movi_interleave_accepts_single_pair() {
        let chunks = [ChunkKind::Video, ChunkKind::Audio];
        validate_movi_interleave(&chunks).expect("single video+audio pair must pass");
    }

    /// §4: device-profile interleave (3 video/audio pairs alternating)
    /// must be accepted. Mirrors the §4 worked-example pattern.
    #[test]
    fn validate_movi_interleave_accepts_three_pairs() {
        let chunks = [
            ChunkKind::Video,
            ChunkKind::Audio,
            ChunkKind::Video,
            ChunkKind::Audio,
            ChunkKind::Video,
            ChunkKind::Audio,
        ];
        validate_movi_interleave(&chunks).expect("3-pair alternation must pass");
    }

    /// §4: first chunk being audio (not video-first) is rejected at
    /// position 0.
    #[test]
    fn validate_movi_interleave_rejects_audio_first() {
        let chunks = [ChunkKind::Audio, ChunkKind::Video];
        let err = validate_movi_interleave(&chunks).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("must be video") && msg.contains("#0"),
            "first-chunk-is-audio error must name #0 and 'must be video', got {msg}"
        );
    }

    /// §4: two consecutive videos (missing audio between them) rejected
    /// at the second video's position.
    #[test]
    fn validate_movi_interleave_rejects_consecutive_videos() {
        let chunks = [ChunkKind::Video, ChunkKind::Video, ChunkKind::Audio];
        let err = validate_movi_interleave(&chunks).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("must be audio") && msg.contains("#1"),
            "consecutive-videos must name #1 and 'must be audio', got {msg}"
        );
    }

    /// §4: two consecutive audios (missing video between them) rejected
    /// at the second audio's position.
    #[test]
    fn validate_movi_interleave_rejects_consecutive_audios() {
        let chunks = [
            ChunkKind::Video,
            ChunkKind::Audio,
            ChunkKind::Audio,
            ChunkKind::Video,
        ];
        let err = validate_movi_interleave(&chunks).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("must be video") && msg.contains("#2"),
            "consecutive-audios must name #2 and 'must be video', got {msg}"
        );
    }

    /// §4: odd-length sequence ending on a video chunk is rejected for
    /// the missing trailing audio block (every video must pair with one
    /// audio per §4 strict 1:1 rule).
    #[test]
    fn validate_movi_interleave_rejects_trailing_unpaired_video() {
        let chunks = [ChunkKind::Video, ChunkKind::Audio, ChunkKind::Video];
        let err = validate_movi_interleave(&chunks).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("must be even") && msg.contains("missing trailing audio"),
            "odd-length must report missing trailing audio, got {msg}"
        );
    }

    /// §4: an `Other` chunk-kind (anything outside the observed
    /// `00dc` / `01wb` set) is rejected at its position with the tag
    /// bytes reported.
    #[test]
    fn validate_movi_interleave_rejects_other_tag() {
        let chunks = [
            ChunkKind::Video,
            ChunkKind::Audio,
            ChunkKind::Other(*b"junk"),
        ];
        let err = validate_movi_interleave(&chunks).unwrap_err();
        let msg = format!("{err:?}");
        // The tag bytes 'j','u','n','k' = 0x6A 0x75 0x6E 0x6B.
        assert!(
            msg.contains("#2") && msg.contains("6A") && msg.contains("75"),
            "Other-tag must name #2 and report tag bytes, got {msg}"
        );
    }

    /// §4: 2232-chunk sequence (comedian.amv's 1116 + 1116 paired
    /// chunks) under the strict video-first alternation must be
    /// accepted. Mirrors the trace's worked example end-to-end.
    #[test]
    fn validate_movi_interleave_accepts_comedian_chunk_count() {
        let mut chunks = Vec::with_capacity(2232);
        for _ in 0..1116 {
            chunks.push(ChunkKind::Video);
            chunks.push(ChunkKind::Audio);
        }
        validate_movi_interleave(&chunks).expect("comedian.amv 1116-pair walk must satisfy §4");
    }

    // ─────────────────── MoviPayloadIter tests ───────────────────

    /// Helper: build a §4 leaf chunk (8-byte header + body) into a
    /// growable buffer. `tag` is the FOURCC; `body` is the raw payload.
    fn push_leaf(out: &mut Vec<u8>, tag: &[u8; 4], body: &[u8]) {
        out.extend_from_slice(tag);
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(body);
    }

    /// Helper: assemble a synthetic `00dc` JPEG-shaped payload with the
    /// requested entropy-byte count between the SOI / EOI bracket.
    fn make_video_body(entropy_len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(4 + entropy_len);
        v.extend_from_slice(&JPEG_SOI);
        v.extend(std::iter::repeat_n(0x00u8, entropy_len));
        v.extend_from_slice(&JPEG_EOI);
        v
    }

    /// Helper: assemble a synthetic `01wb` audio payload with the given
    /// preamble fields and `compressed_len` bytes of (zeroed) body.
    fn make_audio_body(state: u32, decoded_sample_count: u32, compressed_len: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(AMV_AUDIO_PREAMBLE_LEN + compressed_len);
        v.extend_from_slice(&state.to_le_bytes());
        v.extend_from_slice(&decoded_sample_count.to_le_bytes());
        v.extend(std::iter::repeat_n(0x00u8, compressed_len));
        v
    }

    /// An empty `movi` body produces no chunks and no errors.
    #[test]
    fn movi_iter_empty_buffer_yields_no_chunks() {
        let mut it = MoviPayloadIter::new(&[]);
        assert!(it.next().is_none(), "empty input must yield None");
        assert!(it.next().is_none(), "subsequent next() stays None");
        assert_eq!(it.cursor(), 0);
    }

    /// §4: a single `00dc` + `01wb` pair walks cleanly under the strict
    /// 8-byte-header + no-padding rule. Cursors must advance by
    /// `8 + size` per chunk.
    #[test]
    fn movi_iter_walks_single_video_audio_pair() {
        let video = make_video_body(10);
        let audio = make_audio_body(0, 1837, 919);
        let mut buf = Vec::new();
        push_leaf(&mut buf, &VIDEO_CHUNK_TAG, &video);
        push_leaf(&mut buf, &AUDIO_CHUNK_TAG, &audio);
        let mut it = MoviPayloadIter::new(&buf);

        // 1) Video chunk.
        let v = it.next().expect("first chunk").expect("video parse");
        match v {
            MoviPayload::Video { chunk_offset, body } => {
                assert_eq!(chunk_offset, 0);
                assert_eq!(body, video.as_slice());
            }
            other => panic!("expected Video, got {other:?}"),
        }
        assert_eq!(v.kind(), ChunkKind::Video);

        // 2) Audio chunk — typed preamble surfaces alongside body.
        let a = it.next().expect("second chunk").expect("audio parse");
        match a {
            MoviPayload::Audio {
                chunk_offset,
                preamble,
                body,
            } => {
                assert_eq!(chunk_offset, 8 + video.len());
                assert_eq!(preamble.state, 0);
                assert_eq!(preamble.decoded_sample_count, 1837);
                assert_eq!(body, audio.as_slice());
            }
            other => panic!("expected Audio, got {other:?}"),
        }
        assert_eq!(a.kind(), ChunkKind::Audio);

        // 3) Clean end-of-buffer landing.
        assert!(it.next().is_none(), "iterator must terminate cleanly");
        assert_eq!(it.cursor(), buf.len());
    }

    /// §4 no-padding rule: an odd-sized payload doesn't trip the walker
    /// — the cursor advances by `8 + size` even when `size` is odd.
    /// Mirrors the trace's worked example where chunk #2 starts at the
    /// odd file offset `0x07A5`.
    #[test]
    fn movi_iter_no_word_padding_on_odd_sized_payload() {
        // Video body length = 4 + 3 = 7 (odd).
        let v_body = make_video_body(3);
        assert_eq!(v_body.len() % 2, 1);
        // Audio body length = 8 + 1 = 9 (odd).
        let a_body = make_audio_body(0, 100, 1);
        assert_eq!(a_body.len() % 2, 1);

        let mut buf = Vec::new();
        push_leaf(&mut buf, &VIDEO_CHUNK_TAG, &v_body);
        push_leaf(&mut buf, &AUDIO_CHUNK_TAG, &a_body);

        let mut it = MoviPayloadIter::new(&buf);
        let v = it.next().unwrap().unwrap();
        assert_eq!(v.chunk_offset(), 0);
        let a = it.next().unwrap().unwrap();
        // Second chunk header lands at exactly 8 + 7 = 15 (no padding).
        assert_eq!(a.chunk_offset(), 8 + v_body.len());
        assert!(it.next().is_none());
    }

    /// An unknown FOURCC surfaces as the `Other` variant carrying the
    /// tag bytes verbatim. The trace records `00dc` / `01wb` as the
    /// only observed tags, but the walker is permissive so a corrupt /
    /// experimental chunk surfaces as data rather than aborting.
    #[test]
    fn movi_iter_unknown_tag_surfaces_as_other() {
        let mut buf = Vec::new();
        push_leaf(&mut buf, b"junk", &[0xAA, 0xBB, 0xCC]);
        let mut it = MoviPayloadIter::new(&buf);
        let p = it.next().unwrap().unwrap();
        match p {
            MoviPayload::Other {
                chunk_offset,
                tag,
                body,
            } => {
                assert_eq!(chunk_offset, 0);
                assert_eq!(&tag, b"junk");
                assert_eq!(body, &[0xAA, 0xBB, 0xCC]);
            }
            other => panic!("expected Other, got {other:?}"),
        }
        assert_eq!(p.kind(), ChunkKind::Other(*b"junk"));
    }

    /// A chunk-header read that lands in the last `<8` bytes of the
    /// buffer surfaces as a truncation error once, then iteration
    /// terminates (latched-done flag).
    #[test]
    fn movi_iter_trailing_truncation_surfaces_error_once_then_none() {
        let mut buf = Vec::new();
        push_leaf(&mut buf, &VIDEO_CHUNK_TAG, &make_video_body(2));
        // Append 3 trailing bytes — not enough for an 8-byte header.
        buf.extend_from_slice(&[0xDE, 0xAD, 0xBE]);
        let mut it = MoviPayloadIter::new(&buf);
        // First chunk reads cleanly.
        assert!(matches!(
            it.next().unwrap().unwrap(),
            MoviPayload::Video { .. }
        ));
        // Second next() surfaces the truncation error.
        let err = it.next().unwrap().unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(
                    msg.contains("truncated") && msg.contains("3 bytes"),
                    "truncation message must name offset + remaining bytes, got {msg}"
                );
            }
            other => panic!("expected InvalidData, got {other:?}"),
        }
        // Latched-done — no further error replays.
        assert!(it.next().is_none());
        assert!(it.next().is_none());
    }

    /// A chunk whose declared body size runs past the buffer end
    /// surfaces as InvalidData naming the offending offset and the
    /// over-long body end.
    #[test]
    fn movi_iter_body_size_overrun_surfaces_error() {
        let mut buf = Vec::new();
        // Hand-write a chunk header claiming a 100-byte body but
        // provide only 4 body bytes.
        buf.extend_from_slice(&VIDEO_CHUNK_TAG);
        buf.extend_from_slice(&100u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]);
        let mut it = MoviPayloadIter::new(&buf);
        let err = it.next().unwrap().unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(
                    msg.contains("declared size 100") && msg.contains("runs past buffer end"),
                    "overrun message must name size + condition, got {msg}"
                );
            }
            other => panic!("expected InvalidData, got {other:?}"),
        }
        assert!(it.next().is_none());
    }

    /// An `01wb` chunk whose body is shorter than the 8-byte preamble
    /// surfaces as the preamble parser's InvalidData (single-shot, then
    /// iteration terminates).
    #[test]
    fn movi_iter_audio_payload_shorter_than_preamble_errors() {
        let mut buf = Vec::new();
        // 5-byte body — shorter than AMV_AUDIO_PREAMBLE_LEN = 8.
        push_leaf(&mut buf, &AUDIO_CHUNK_TAG, &[0, 0, 0, 0, 0]);
        let mut it = MoviPayloadIter::new(&buf);
        let err = it.next().unwrap().unwrap_err();
        match err {
            AmvDemuxerError::InvalidData(msg) => {
                assert!(
                    msg.contains("preamble") && msg.contains("got 5"),
                    "preamble-too-short message must name shortfall, got {msg}"
                );
            }
            other => panic!("expected InvalidData, got {other:?}"),
        }
        assert!(it.next().is_none());
    }

    /// §4 worked example shape: a 3-pair (6-chunk) walk emits the
    /// strict video-first alternation, and the kinds collected from the
    /// iterator pass [`validate_movi_interleave`].
    #[test]
    fn movi_iter_three_pair_walk_kinds_pass_strict_interleave() {
        let mut buf = Vec::new();
        for i in 0..3 {
            // Vary sizes slightly per pair so we're not just walking a
            // single repeating record.
            push_leaf(&mut buf, &VIDEO_CHUNK_TAG, &make_video_body(10 + i));
            push_leaf(&mut buf, &AUDIO_CHUNK_TAG, &make_audio_body(0, 100, 5));
        }
        let it = MoviPayloadIter::new(&buf);
        let kinds: Vec<ChunkKind> = it.map(|r| r.expect("clean 3-pair walk").kind()).collect();
        assert_eq!(kinds.len(), 6);
        validate_movi_interleave(&kinds).expect("3 pairs must satisfy §4");
    }

    /// Audio preamble accessors round-trip through the iterator —
    /// every emitted `Audio` variant's preamble matches what
    /// [`AmvAudioPreamble::parse`] would return on the same body.
    #[test]
    fn movi_iter_audio_preamble_matches_independent_parse() {
        let body = make_audio_body(0xDEADBEEF, 1837, 919);
        let mut buf = Vec::new();
        push_leaf(&mut buf, &AUDIO_CHUNK_TAG, &body);
        let mut it = MoviPayloadIter::new(&buf);
        let p = it.next().unwrap().unwrap();
        let independent = AmvAudioPreamble::parse(&body).unwrap();
        match p {
            MoviPayload::Audio { preamble, .. } => assert_eq!(preamble, independent),
            other => panic!("expected Audio, got {other:?}"),
        }
    }

    /// `body()` accessor returns the same slice as the variant's body
    /// field — confirms the convenience method doesn't re-slice or
    /// truncate.
    #[test]
    fn movi_iter_body_accessor_matches_payload_field() {
        let v_body = make_video_body(7);
        let a_body = make_audio_body(0, 200, 50);
        let mut buf = Vec::new();
        push_leaf(&mut buf, &VIDEO_CHUNK_TAG, &v_body);
        push_leaf(&mut buf, &AUDIO_CHUNK_TAG, &a_body);
        let it = MoviPayloadIter::new(&buf);
        let payloads: Vec<MoviPayload> = it.map(|r| r.unwrap()).collect();
        assert_eq!(payloads[0].body(), v_body.as_slice());
        assert_eq!(payloads[1].body(), a_body.as_slice());
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
