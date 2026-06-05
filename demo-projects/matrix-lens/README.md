# Matrix Lens

Desktop capture lens with matrix, glass, thermal, and xray modes over live or
pattern input.

```sh
cargo run -- --capture auto --resolution 720p --mode matrix
```

Useful options include `--frames`, `--output`, `--capture`, `--mode`,
`--resolution`, and `--fps-limit`. This demo depends on desktop capture,
PipeWire/Wayland paths, Vulkan presentation, and ROCm device execution.
