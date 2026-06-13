//! Bench: write a full AMV file through the public `AmvMuxer` —
//! `write_header` (§1 prelude + zeroed sizes + §2 `amvh` + §3 stream
//! headers + audio `WAVEFORMATEX`), `write_packet` for 1116 video-first
//! `00dc`/`01wb` pairs (§4 no-padding chunk emission + frame-count
//! accumulation), and `write_trailer` (§2 packed-duration patch +
//! §4c `AMV_END_`). This is the container write hot path.
//!
//! Run with: `cargo bench -p oxideav-amv --bench mux_write`

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

#[path = "common/mod.rs"]
mod common;

fn bench_mux_write(c: &mut Criterion) {
    const N_PAIRS: usize = 1116;

    let mut group = c.benchmark_group("mux_write");
    group.throughput(Throughput::Elements((N_PAIRS * 2) as u64));
    group.bench_function("comedian_1116_pairs", |b| {
        b.iter(|| {
            let bytes = common::build_amv_bytes(black_box(N_PAIRS));
            black_box(bytes.len())
        });
    });
    group.finish();
}

criterion_group!(benches, bench_mux_write);
criterion_main!(benches);
