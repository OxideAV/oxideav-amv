//! Bench: the §4a rate-controlled frame encode vs the unconstrained
//! encode — how much the rate–distortion budget search costs on top of
//! a plain encode.
//!
//! The budgeted path quantizes once (DCT + quant) and then bisects the
//! Lagrangian price λ, re-running the per-block RD dynamic program plus
//! an allocation-free exact-size counting walk per probe (bracket +
//! ≤ 12 bisections, with an early exit once the plan uses ≥ 99.5 % of
//! the budget). The binding cases therefore cost a couple dozen entropy-
//! stage passes — still a few percent of the 83 ms frame interval of
//! the §2 12 fps device profile. Cases on the same deterministic
//! textured 128×96 frame (the comedian geometry):
//!
//! * `unconstrained` — the plain `encode_frame_rgb` baseline;
//! * `budget_unbinding` — a budget the full encode already fits (one
//!   entropy pass, no search);
//! * `budget_half` / `budget_fifth` — binding budgets that engage the
//!   full binary search.
//!
//! Run with: `cargo bench -p oxideav-amv --bench rate_control_encode`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use oxideav_amv::{encode_frame_rgb, encode_frame_rgb_with_budget};

/// Deterministic textured RGB frame (gradients + LCG noise) with real
/// AC energy, mirroring the unit-test content generator.
fn textured_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    let mut lcg: u32 = 0x1234_5678;
    for y in 0..h {
        for x in 0..w {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (lcg >> 24) as i32 - 128;
            let k = ((y * w + x) * 3) as usize;
            let base = (x * 2 + y * 3) as i32;
            rgb[k] = (base % 200 + 28 + noise / 4).clamp(0, 255) as u8;
            rgb[k + 1] = ((base * 2) % 180 + 40 + noise / 6).clamp(0, 255) as u8;
            rgb[k + 2] = ((x + y) as i32 % 160 + 48 + noise / 8).clamp(0, 255) as u8;
        }
    }
    rgb
}

fn bench_rate_control_encode(c: &mut Criterion) {
    const W: u32 = 128;
    const H: u32 = 96;
    let rgb = textured_rgb(W, H);
    let full_len = encode_frame_rgb(W, H, &rgb).expect("baseline encode").len();

    let mut group = c.benchmark_group("rate_control_encode");
    group.bench_function("unconstrained", |b| {
        b.iter(|| {
            let payload = encode_frame_rgb(black_box(W), black_box(H), black_box(&rgb)).unwrap();
            black_box(payload.len())
        });
    });
    group.bench_function("budget_unbinding", |b| {
        b.iter(|| {
            let f = encode_frame_rgb_with_budget(
                black_box(W),
                black_box(H),
                black_box(&rgb),
                black_box(full_len),
            )
            .unwrap();
            black_box(f.payload.len())
        });
    });
    group.bench_function("budget_half", |b| {
        b.iter(|| {
            let f = encode_frame_rgb_with_budget(
                black_box(W),
                black_box(H),
                black_box(&rgb),
                black_box(full_len / 2),
            )
            .unwrap();
            black_box(f.payload.len())
        });
    });
    group.bench_function("budget_fifth", |b| {
        b.iter(|| {
            let f = encode_frame_rgb_with_budget(
                black_box(W),
                black_box(H),
                black_box(&rgb),
                black_box(full_len / 5),
            )
            .unwrap();
            black_box(f.payload.len())
        });
    });
    // Streaming steady state: the λ warm start the registry encoder
    // carries across frames — seed each encode with the fitted price of
    // an identical previous frame (the search should collapse to a
    // probe or two instead of the full bracket + bisection).
    {
        use oxideav_amv::{
            decode_frame_yuv420p_from_payload, encode_frame_yuv420p_with_budget_seeded, AmvHeader,
        };
        let header = AmvHeader {
            micros_per_frame: 83_333,
            width: W,
            height: H,
            fps: 12,
            flag_one: 1,
            reserved_30: 0,
            duration_packed: 0,
        };
        let full = encode_frame_rgb(W, H, &rgb).expect("encode");
        let yuv = decode_frame_yuv420p_from_payload(&header, &full).expect("planes");
        let budget = full_len / 2;
        let (_f, seed) =
            encode_frame_yuv420p_with_budget_seeded(W, H, &yuv.y, &yuv.cb, &yuv.cr, budget, None)
                .expect("seed encode");
        assert!(seed.is_some());
        group.bench_function("budget_half_warm_started", |b| {
            b.iter(|| {
                let (f, l) = encode_frame_yuv420p_with_budget_seeded(
                    black_box(W),
                    black_box(H),
                    black_box(&yuv.y),
                    black_box(&yuv.cb),
                    black_box(&yuv.cr),
                    black_box(budget),
                    black_box(seed),
                )
                .unwrap();
                black_box((f.payload.len(), l))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_rate_control_encode);
criterion_main!(benches);
