//! Bench: build the in-memory chunk index over a whole AMV file via
//! `AmvDemuxer::build_chunk_index`. This is the up-front
//! Seek-skip-bodies walk that records every chunk's file offset plus
//! the per-stream pre-emit PTS, the table that lets repeated random
//! seeks short-circuit the linear walk. It exercises the same
//! chunk-header parse + §4 no-padding advance as the drain path but
//! skips payload reads (`Seek` past each body), so the A/B vs.
//! `demux_drain` isolates the body-read cost.
//!
//! Run with: `cargo bench -p oxideav-amv --bench build_index`

use std::io::Cursor;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use oxideav_amv::AmvDemuxer;

#[path = "common/mod.rs"]
mod common;

fn bench_build_index(c: &mut Criterion) {
    const N_PAIRS: usize = 1116;
    let bytes = common::build_amv_bytes(N_PAIRS);

    let mut group = c.benchmark_group("build_index");
    group.throughput(Throughput::Elements((N_PAIRS * 2) as u64));
    group.bench_function("comedian_1116_pairs", |b| {
        b.iter(|| {
            let mut d = AmvDemuxer::open(Cursor::new(bytes.clone())).expect("open");
            d.build_chunk_index().expect("build_chunk_index");
            black_box(d.chunk_index().map(|i| i.len()))
        });
    });
    group.finish();
}

criterion_group!(benches, bench_build_index);
criterion_main!(benches);
