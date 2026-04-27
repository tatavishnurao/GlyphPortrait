# GlyphForge GPU Roadmap

The current core pipeline is local-first and CPU-friendly. GPU usage today is
primarily optional metrics visibility and future extension points.

## Stage 1: GPU segmentation

- Add ONNX Runtime CUDA and/or Torch segmentation backends.
- Keep CPU fallback path intact for portability.
- Expose backend selection in Gradio and CLI.

## Stage 2: GPU upscaling

- Add optional super-resolution path for high-resolution poster export.
- Support ESRGAN/Real-ESRGAN style local models.

## Stage 3: Local diffusion/img2img stylization

- Add optional post-process stylization pass using local diffusion models.
- Keep typography-first rendering as the primary source image.

## Stage 4: Guided stylization (ControlNet-style)

- Add structure-preserving stylization modes that respect mask and portrait form.
- Make stylization optional and reproducible with seed control.

## Stage 5: Benchmark and profiling panel

- Add benchmark panel for:
  - render time
  - VRAM usage
  - output resolution
  - words placed and attempts
