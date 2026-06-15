#![no_main]

//! Drive arbitrary fuzz-supplied bytes through the full
//! [`oxideav_amv::AmvDemuxer::open`] + bounded `next_packet` drain
//! path.
//!
//! Where `parse.rs` exercises every public byte parser in isolation,
//! this target stitches them together exactly the way the demuxer
//! does: a §1 RIFF probe, a §2 `amvh` body read, a §3 `strl` LIST
//! walk for both streams (video first), a §3b audio `WAVEFORMATEX`
//! decode, then the §4 `movi` payload loop where each iteration reads
//! an 8-byte chunk header, advances the cursor by `8 + size` (no
//! word padding — §4 "no-padding rule"), and stamps a per-packet PTS
//! from the §2 `fps` (video) or the §4b preamble's
//! `decoded_sample_count` (audio). The §4c `AMV_END_` ASCII trailer
//! terminates the walk cleanly, and any short read at a chunk
//! boundary is treated as a graceful truncation per the device-cut
//! recovery contract — both signals flow through the same loop.
//!
//! Contract under test: `AmvDemuxer::open(bytes)` ALWAYS returns a
//! `Result<…, AmvDemuxerError>` — no panic, no abort, no integer
//! overflow (debug build), no out-of-bounds index, no allocation
//! proportional to an attacker-controlled `size` field in a chunk
//! header. Every `next_packet` call on a successfully-opened demuxer
//! likewise returns `Ok(Packet)` or `Err(Error)`. The strict-open
//! [`oxideav_amv::AmvDemuxer::open_strict`] entry point is also
//! driven on every input — its §2 + §3 sentinel checks live in the
//! parse path, so a fuzz input that passes the permissive open may
//! still trip strict's cross-checks; either branch must return
//! gracefully.
//!
//! The drain loop is capped at 32 packets per fuzz iteration so a
//! pathological input that emits one tiny chunk per iteration cannot
//! starve the fuzzer's budget. The cap is exercised against the
//! staged 2232-chunk comedian fixture in the unit-test suite, not
//! here — this target is panic-free-only by design and the return
//! values are intentionally discarded.

use libfuzzer_sys::fuzz_target;
use oxideav_amv::AmvDemuxer;
use std::io::Cursor;

fuzz_target!(|data: &[u8]| {
    // -----------------------------------------------------------------
    // Permissive open. The vast majority of fuzz inputs fall out at
    // the §1 RIFF probe (`RIFF .... 'AMV '`) or the §2 `amvh` length
    // check; these are the cheapest reject paths and don't reach the
    // chunk loop. Inputs that do pass the prelude pass into the
    // bounded drain below.
    // -----------------------------------------------------------------
    let cursor = Cursor::new(data.to_vec());
    if let Ok(mut demux) = AmvDemuxer::open(cursor) {
        // The opened demuxer's header / audio_format accessors are
        // infallible — drive them once per opened input so a
        // regression that panics in a `Debug` formatter or in a
        // duration helper is caught here.
        let _ = format!("{:?}", demux.header());
        let _ = format!("{:?}", demux.audio_format());
        let _ = demux.is_truncated();

        // Bounded drain of the §4 `movi` payload. 32 iterations
        // covers every shape the trace records (interleaved 1:1
        // video-first, mid-walk truncation, clean trailer) without
        // letting an attacker run the fuzzer until budget exhaustion.
        for _ in 0..32 {
            // Using the framework `Demuxer` trait would require
            // bringing the trait into scope; the inherent
            // `next_packet` is not public, so we instead exercise
            // `AmvDemuxer`'s public sibling — the `Demuxer` trait
            // path is via `oxideav_core::Demuxer`.
            use oxideav_core::Demuxer;
            match demux.next_packet() {
                Ok(_pkt) => {}
                Err(_e) => break,
            }
        }
    }

    // -----------------------------------------------------------------
    // Strict open. Same input, stricter prelude validation — the §2
    // and §3 sentinel checks run before the cursor lands on `movi`.
    // Most inputs that pass permissive open still fail here (zero
    // `flag_one`, non-zero `reserved_30`, mismatched
    // `dwMicroSecPerFrame`, non-zero stream-header bodies, ...) and
    // that's the entire point of the strict variant; the contract is
    // still "every path returns without panicking".
    // -----------------------------------------------------------------
    let cursor = Cursor::new(data.to_vec());
    let _ = AmvDemuxer::open_strict(cursor);
});
