//! Stream-level rate control for the AMV video track.
//!
//! [`AmvRateController`] turns a stream-wide target (a video bitrate, or
//! equivalently a mean payload size per frame) into a **per-frame byte
//! budget** for the §4a budgeted frame encode
//! ([`crate::encode_frame_rgb_with_budget`] /
//! [`crate::encode_frame_yuv420p_with_budget`]), using a carry account:
//! frames that undershoot their budget (simple content) donate their
//! unspent bytes to later frames, and frames that had to overshoot
//! (their DC-only floor exceeded the budget) borrow against later
//! frames. The carry is clamped so a long run of trivial frames cannot
//! hoard an unbounded byte reserve and one pathological frame cannot
//! starve the rest of the stream.
//!
//! # Why the budget is on the *payload*, not the file
//!
//! AMV's container overhead per video frame is a constant 8-byte leaf
//! chunk header (§4 chunk framing), so a container-level bitrate target
//! differs from the payload-level one by exactly `8 × fps` bytes/s; the
//! controller works in payload bytes (what the budgeted encode
//! controls) and leaves the fixed framing arithmetic to the caller.
//! Audio has **no rate headroom to control**: the §4b IMA-ADPCM profile
//! is format-fixed at 4 bits per sample plus the 8-byte block preamble,
//! so the audio track's rate is fully determined by the sample rate and
//! fps (e.g. 22 050 Hz ÷ 12 fps → 1837/1838 samples → 927/928-byte
//! blocks). Rate control is therefore a video-only concern in AMV.
//!
//! The controller is deliberately encode-agnostic — it only trades
//! budgets against measured sizes — so it is equally usable by callers
//! driving `encode_frame_*_with_budget` by hand and by the registered
//! `amv_video` [`Encoder`](oxideav_core::Encoder), which wires one up
//! automatically when `CodecParameters::bit_rate` is set.

use crate::AmvDemuxerError;

/// Hard lower bound on any per-frame budget the controller hands out,
/// in bytes. Keeps the budget request meaningful (a bare §4a payload is
/// at minimum `FF D8` + entropy + `FF D9`) even when the carry account
/// is deeply overdrawn; the budgeted encode itself enforces the true
/// content-dependent DC-only floor.
const MIN_FRAME_BUDGET: u64 = 8;

/// How many frames' worth of target bytes the carry account may hold
/// (in either direction). Bounds the burst a donated reserve can fund
/// and the squeeze an overdraft can inflict, keeping the delivered rate
/// close to the target over any window a device buffer would care
/// about.
const CARRY_CAP_FRAMES: u64 = 4;

/// Stream-level rate controller for AMV `00dc` video payloads: hands
/// out one byte budget per frame and books the bytes actually spent.
///
/// ```
/// use oxideav_amv::{encode_frame_rgb_with_budget, AmvRateController};
///
/// // 96 kbit/s of video payload at the comedian profile's 12 fps
/// // → 1000 bytes/frame.
/// let mut rc = AmvRateController::from_video_bitrate(96_000, 12).unwrap();
/// assert_eq!(rc.target_bytes_per_frame(), 1000);
/// let rgb = vec![128u8; 128 * 96 * 3];
/// for _ in 0..3 {
///     let budget = rc.frame_budget();
///     let frame = encode_frame_rgb_with_budget(128, 96, &rgb, budget).unwrap();
///     rc.note_frame(frame.payload.len());
/// }
/// assert!(rc.average_bytes_per_frame() <= 1000.0);
/// ```
#[derive(Debug, Clone)]
pub struct AmvRateController {
    /// Long-run mean payload size the stream must hold, bytes/frame.
    target_bytes_per_frame: u64,
    /// Byte credit (+) or debt (−) accumulated against the target,
    /// clamped to ±[`CARRY_CAP_FRAMES`] × target.
    carry: i64,
    /// Frames booked via [`Self::note_frame`].
    frames: u64,
    /// Total payload bytes booked via [`Self::note_frame`].
    total_payload_bytes: u64,
}

impl AmvRateController {
    /// Build a controller from a mean **payload** size per frame.
    /// Rejects a zero target (no encode can average zero bytes).
    pub fn from_bytes_per_frame(target_bytes_per_frame: u64) -> Result<Self, AmvDemuxerError> {
        if target_bytes_per_frame == 0 {
            return Err(AmvDemuxerError::InvalidData(
                "amv rate control: target bytes/frame must be non-zero".into(),
            ));
        }
        Ok(Self {
            target_bytes_per_frame,
            carry: 0,
            frames: 0,
            total_payload_bytes: 0,
        })
    }

    /// Build a controller from a video **payload bitrate** (bits per
    /// second) and the §2 integer frame rate: the per-frame target is
    /// `bits_per_sec / 8 / fps`. Rejects a zero rate on either side and
    /// a combination that rounds down to a zero-byte frame target.
    pub fn from_video_bitrate(bits_per_sec: u64, fps: u32) -> Result<Self, AmvDemuxerError> {
        if bits_per_sec == 0 || fps == 0 {
            return Err(AmvDemuxerError::InvalidData(format!(
                "amv rate control: bitrate and fps must be non-zero (got {bits_per_sec} bps @ {fps} fps)",
            )));
        }
        let target = bits_per_sec / 8 / fps as u64;
        if target == 0 {
            return Err(AmvDemuxerError::InvalidData(format!(
                "amv rate control: {bits_per_sec} bps @ {fps} fps rounds to a zero-byte frame target",
            )));
        }
        Self::from_bytes_per_frame(target)
    }

    /// The long-run mean payload size this controller holds the stream
    /// to, in bytes per frame.
    pub fn target_bytes_per_frame(&self) -> u64 {
        self.target_bytes_per_frame
    }

    /// The byte budget for the **next** frame: the per-frame target
    /// plus the current carry credit (or minus the current debt), never
    /// below a small hard floor. Pass this to
    /// [`crate::encode_frame_rgb_with_budget`] /
    /// [`crate::encode_frame_yuv420p_with_budget`], then book the
    /// resulting payload with [`Self::note_frame`].
    pub fn frame_budget(&self) -> usize {
        let budget = self.target_bytes_per_frame as i64 + self.carry;
        budget.max(MIN_FRAME_BUDGET as i64) as usize
    }

    /// Book the payload size a frame actually spent. Unspent budget
    /// (relative to the per-frame *target*, not the handed-out budget)
    /// becomes carry credit for later frames; overspend becomes debt.
    /// The carry saturates at ±[`CARRY_CAP_FRAMES`] frames' worth of
    /// target bytes.
    pub fn note_frame(&mut self, payload_bytes: usize) {
        let cap = (self.target_bytes_per_frame * CARRY_CAP_FRAMES) as i64;
        self.carry = (self.carry + self.target_bytes_per_frame as i64 - payload_bytes as i64)
            .clamp(-cap, cap);
        self.frames += 1;
        self.total_payload_bytes += payload_bytes as u64;
    }

    /// Frames booked so far.
    pub fn frames_noted(&self) -> u64 {
        self.frames
    }

    /// Total payload bytes booked so far.
    pub fn total_payload_bytes(&self) -> u64 {
        self.total_payload_bytes
    }

    /// Mean booked payload size, bytes per frame (`0.0` before any
    /// frame is booked).
    pub fn average_bytes_per_frame(&self) -> f64 {
        if self.frames == 0 {
            return 0.0;
        }
        self.total_payload_bytes as f64 / self.frames as f64
    }

    /// Mean delivered video payload bitrate in bits per second at the
    /// §2 integer frame rate (`0.0` before any frame is booked).
    pub fn achieved_bits_per_sec(&self, fps: u32) -> f64 {
        self.average_bytes_per_frame() * 8.0 * fps as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_reject_zero_targets() {
        assert!(AmvRateController::from_bytes_per_frame(0).is_err());
        assert!(AmvRateController::from_video_bitrate(0, 12).is_err());
        assert!(AmvRateController::from_video_bitrate(96_000, 0).is_err());
        // 8 bps @ 12 fps → 0 bytes/frame.
        assert!(AmvRateController::from_video_bitrate(8, 12).is_err());
    }

    #[test]
    fn bitrate_constructor_matches_comedian_profile_arithmetic() {
        // 96 kbit/s @ 12 fps → 96000/8/12 = 1000 bytes/frame.
        let rc = AmvRateController::from_video_bitrate(96_000, 12).unwrap();
        assert_eq!(rc.target_bytes_per_frame(), 1000);
        assert_eq!(rc.frame_budget(), 1000);
        // 16 fps profile: 96000/8/16 = 750.
        let rc = AmvRateController::from_video_bitrate(96_000, 16).unwrap();
        assert_eq!(rc.target_bytes_per_frame(), 750);
    }

    #[test]
    fn undershoot_donates_carry_to_the_next_frame() {
        let mut rc = AmvRateController::from_bytes_per_frame(1000).unwrap();
        rc.note_frame(400); // 600 unspent
        assert_eq!(rc.frame_budget(), 1600);
        rc.note_frame(1600); // spends target + full credit
        assert_eq!(rc.frame_budget(), 1000);
    }

    #[test]
    fn overshoot_borrows_from_later_frames() {
        let mut rc = AmvRateController::from_bytes_per_frame(1000).unwrap();
        // A frame whose DC-only floor exceeded the budget.
        rc.note_frame(1500);
        assert_eq!(rc.frame_budget(), 500);
        rc.note_frame(500);
        assert_eq!(rc.frame_budget(), 1000);
    }

    #[test]
    fn carry_saturates_in_both_directions() {
        let mut rc = AmvRateController::from_bytes_per_frame(100).unwrap();
        // 100 trivial frames: credit would be ~9900 unclamped; the cap
        // holds it at 4 frames' worth.
        for _ in 0..100 {
            rc.note_frame(1);
        }
        assert_eq!(rc.frame_budget(), 100 + 400);
        // A run of forced overshoots: debt clamps at −400, so the
        // budget never collapses below the hard floor.
        for _ in 0..100 {
            rc.note_frame(1000);
        }
        assert_eq!(rc.frame_budget(), MIN_FRAME_BUDGET as usize);
    }

    #[test]
    fn budget_never_falls_below_the_hard_floor() {
        let mut rc = AmvRateController::from_bytes_per_frame(10).unwrap();
        rc.note_frame(10_000); // debt clamps at −40
        assert_eq!(rc.frame_budget(), MIN_FRAME_BUDGET as usize);
    }

    #[test]
    fn long_run_average_holds_the_target_when_floors_allow() {
        // Synthetic encoder: each frame naturally wants `natural` bytes
        // but can compress to any budget ≥ its floor of 60 bytes.
        let mut rc = AmvRateController::from_bytes_per_frame(500).unwrap();
        let mut lcg: u32 = 0xDEAD_BEEF;
        for _ in 0..1000 {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let natural = 200 + (lcg >> 22) as usize; // 200..=1223
            let budget = rc.frame_budget();
            let spent = natural.min(budget).max(60);
            rc.note_frame(spent);
        }
        let avg = rc.average_bytes_per_frame();
        assert!(
            avg <= 500.0,
            "long-run average {avg} must not exceed the 500-byte target"
        );
        assert!(
            avg > 400.0,
            "long-run average {avg} should use most of the target, not starve"
        );
        // And the bitrate view agrees: avg × 8 × fps.
        let bps = rc.achieved_bits_per_sec(12);
        assert!((bps - avg * 96.0).abs() < 1e-6);
    }

    #[test]
    fn stats_start_at_zero() {
        let rc = AmvRateController::from_bytes_per_frame(1000).unwrap();
        assert_eq!(rc.frames_noted(), 0);
        assert_eq!(rc.total_payload_bytes(), 0);
        assert_eq!(rc.average_bytes_per_frame(), 0.0);
        assert_eq!(rc.achieved_bits_per_sec(12), 0.0);
    }
}
