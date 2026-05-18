# oxideav-amv

A pure-Rust AMV (Anychat MV / S1 MP3 / generic Chinese-MP4-player
container + IMA-ADPCM-audio + MJPEG-video) codec for the
[oxideav](https://github.com/OxideAV/oxideav) framework.

## Status

**Orphan-rebuild scaffold (2026-05-18).** The prior implementation was
retired under the workspace
[clean-room policy](https://github.com/OxideAV/oxideav/blob/master/docs/IMPLEMENTOR_ROUND.md):
the demuxer's module-level doc-comment acknowledged that the AMV
chunk-tree layout was learned by reading an external library's
container source — which violates the clean-room provenance
requirement even though AMV has no public written specification.
Master history was fully erased per the Hat-3 cold-enforcement
procedure.

The implementation will be re-built against fresh reverse-engineering
of real AMV file samples (byte traces only, no external library
source) in a future clean-room round.

## License

MIT — see [LICENSE](./LICENSE).
