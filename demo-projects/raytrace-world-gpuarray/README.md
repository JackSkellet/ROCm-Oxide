# Raytrace World GpuArray

Copy of `demo-projects/raytrace-world` rewritten to use the method-oriented
`GpuArray<T>` host helper for frame and camera device buffers.

```sh
cargo run -- --present vulkan --frames 300
```

Controls include WASD movement, arrows for look direction, Shift for speed, `1`
for shadows, `2` for reflections, Space pause, and `R` reset.

## Size Comparison

The first `GpuArray<T>` rewrite changes the host buffer API without changing
the generated raytrace kernel or the local presenter. The Rust source line count
is therefore unchanged:

```text
Original raytrace-world Rust source:      1,958 lines
GpuArray copy Rust source:                1,958 lines
Delta:                                        0 lines
```

Reproduce the measurement with:

```sh
wc -l ../raytrace-world/src/main.rs \
  ../raytrace-world/src/visual_presenter.rs \
  src/main.rs \
  src/visual_presenter.rs
```
