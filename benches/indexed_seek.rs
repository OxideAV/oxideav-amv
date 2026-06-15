//! Bench: indexed random-access seek. Build the chunk index once, then
//! issue a spread of `Demuxer::seek_to` calls across the file. With the
//! index populated, `seek_to` runs the binary-search-backed
//! `seek_to_via_index` path instead of re-reading every chunk header
//! per seek, so this measures the index lookup + single-chunk
//! repositioning cost. A companion group seeks on the same file
//! **without** building the index first, exercising the linear
//! disk-walking fallback for an A/B on the index payoff.
//!
//! Run with: `cargo bench -p oxideav-amv --bench indexed_seek`

use std::io::Cursor;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use oxideav_core::Demuxer;

use oxideav_amv::AmvDemuxer;

#[path = "common/mod.rs"]
mod common;

/// A spread of video-stream PTS targets across the clip: the video
/// time_base is 1/12 so a PTS equals the frame index. Mix forward and
/// backward jumps so both the rewind-to-`movi_start` and the
/// walk-forward branches are hit.
fn seek_targets(n_pairs: usize) -> Vec<i64> {
    let n = n_pairs as i64;
    vec![
        n / 2,
        n / 4,
        (n * 3) / 4,
        10,
        n - 5,
        n / 3,
        (n * 2) / 3,
        1,
        n / 8,
        (n * 7) / 8,
    ]
}

fn bench_indexed_seek(c: &mut Criterion) {
    const N_PAIRS: usize = 1116;
    let bytes = common::build_amv_bytes(N_PAIRS);
    let targets = seek_targets(N_PAIRS);

    let mut group = c.benchmark_group("indexed_seek");
    group.throughput(Throughput::Elements(targets.len() as u64));

    group.bench_function("indexed_1116_pairs", |b| {
        // Build the demuxer + index once per measurement so the seek
        // cost is isolated from index construction (benched separately
        // in build_index).
        b.iter_batched(
            || {
                let mut d = AmvDemuxer::open(Cursor::new(bytes.clone())).expect("open");
                d.build_chunk_index().expect("build_chunk_index");
                d
            },
            |mut d| {
                for &pts in &targets {
                    let landed = d.seek_to(0, pts).expect("seek_to");
                    black_box(landed);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("linear_1116_pairs", |b| {
        b.iter_batched(
            || AmvDemuxer::open(Cursor::new(bytes.clone())).expect("open"),
            |mut d| {
                for &pts in &targets {
                    let landed = d.seek_to(0, pts).expect("seek_to");
                    black_box(landed);
                }
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_indexed_seek);
criterion_main!(benches);
