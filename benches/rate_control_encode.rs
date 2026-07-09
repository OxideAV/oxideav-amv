//! Bench: the §4a rate-controlled frame encode vs the unconstrained
//! encode — how much the budget search costs on top of a plain encode.
//!
//! The budgeted path quantizes once (DCT + quant, the expensive stage)
//! and then binary-searches the trim level over the entropy stage only
//! (≤ 12 cheap passes), so the expected overhead is well under the
//! naive "12× the encode". Three cases on the same deterministic
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
    group.finish();
}

criterion_group!(benches, bench_rate_control_encode);
criterion_main!(benches);
