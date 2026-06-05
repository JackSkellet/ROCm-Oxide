# Window Effects Lab

Captured-window GPU effects pipeline with a control panel and shared Vulkan
presentation.

```sh
cargo run -- --present vulkan --frames 300 0
```

Set `ROCM_OXIDE_WINDOW_FX_TARGET` or pass a positional window selector to choose
the captured window.
