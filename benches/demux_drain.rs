//! Bench: open an AMV file and drain every packet to the `AMV_END_`
//! trailer through the public `AmvDemuxer` + `Demuxer::next_packet`
//! walk. This is the container hot path — 8-byte chunk-header parse,
//! the §4 no-padding `8 + size` advance, per-stream PTS accounting,
//! and trailer-bounded termination — exercised over a 1116-pair file
//! (the `comedian.amv` chunk count).
//!
//! Run with: `cargo bench -p oxideav-amv --bench demux_drain`

use std::io::Cursor;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use oxideav_core::{Demuxer, Error};

use oxideav_amv::AmvDemuxer;

#[path = "common/mod.rs"]
mod common;

fn bench_demux_drain(c: &mut Criterion) {
    const N_PAIRS: usize = 1116;
    let bytes = common::build_amv_bytes(N_PAIRS);

    let mut group = c.benchmark_group("demux_drain");
    // Two chunks (video + audio) per pair; throughput is chunks/s.
    group.throughput(Throughput::Elements((N_PAIRS * 2) as u64));
    group.bench_function("comedian_1116_pairs", |b| {
        b.iter(|| {
            let mut d = AmvDemuxer::open(Cursor::new(bytes.clone())).expect("open");
            let mut n = 0u64;
            loop {
                match d.next_packet() {
                    Ok(p) => {
                        black_box(&p.data);
                        n += 1;
                    }
                    Err(Error::Eof) => break,
                    Err(e) => panic!("walk error: {e:?}"),
                }
            }
            black_box(n)
        });
    });
    group.finish();
}

criterion_group!(benches, bench_demux_drain);
criterion_main!(benches);
