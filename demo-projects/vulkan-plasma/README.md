# Vulkan Plasma Demo

Small Vulkan-presenter smoke test for ROCm-Oxide. The frame is generated on the
CPU, then presented through the same SDL2 + ash Vulkan path used by the visual
examples.

Run from this folder:

```sh
ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run
```

Run a bounded smoke test:

```sh
ROCM_OXIDE_VISUAL_PRESENT=vulkan cargo run -- --frames 300
```

The demo also supports the shared presenter option:

```sh
cargo run -- --present vulkan --frames 300
```

Controls:

- `Esc` closes the window.
- `W` / `S` increase or decrease animation speed.
- `A` / `D` change the pattern scale.
- `Up` / `Down` change color cycling.
