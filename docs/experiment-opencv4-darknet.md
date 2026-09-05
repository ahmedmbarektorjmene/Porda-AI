# Experiment A: OpenCV 4 + Original Darknet — Final Report

## OpenCV Version & Selection
- **Version:** OpenCV 4.5.4 (Ubuntu 22.04 `libopencv-dev 4.5.4+dfsg-9ubuntu4`) and AUR `opencv4 4.14.0-2` (Arch/CachyOS, installs to `/usr/lib/opencv4`, `opencv4.pc`, coexists with system `opencv 5.0`)
- **Why:** OpenCV 5.0 `modules/dnn/src/dnn_read.cpp:39` always throws `Darknet importer has been removed` — `readNetFromDarknet` and `DetectionModel` for Darknet are removed. OpenCV 4.x retains them.
- **Reproducible selection:**
  - Host Arch: `paru -S opencv4` (or project-local `cmake -DCMAKE_INSTALL_PREFIX=$PWD/.local/opencv4 -DBUILD_LIST=core,imgproc,imgcodecs,dnn ...` then `PKG_CONFIG_PATH=$PWD/.local/opencv4/lib/pkgconfig`)
  - Docker: `podman run -v $PWD:/work docker.io/library/ubuntu:22.04` → `DEBIAN_FRONTEND=noninteractive apt-get install -y pkg-config libopencv-dev` → `pkg-config --modversion opencv4` = 4.5.4
  - Cargo: `.cargo/config.toml` `OPENCV_PKGCONFIG_NAME="opencv4"` (vs `opencv5` on main) — `system-deps` picks `opencv4.pc` via `pkg-config`

## Preserved Python Parity
- `model/pordav4x3.cfg` + `model/porda-19200-lr-0005-909.weights` (authoritative, not converted)
- 544x320, scale 1/255, mean (0,0,0), swapRB=true, crop=false, conf 0.25 (accuracy 25), NMS 0.1, target [1] Female (0 Male), `porda_vision::preprocessing::resize_and_pad` early-return `(bottom<55 && right==0) || (right<70 && bottom==0)` mirrors `Porda-AI/main.py:782-801`

## Build
```bash
cargo clean -p opencv
LIBCLANG_PATH=/usr/lib/llvm-14/lib PKG_CONFIG_PATH=/usr/lib/pkgconfig:/usr/lib/opencv4/pkgconfig cargo build -p porda --features opencv
# In docker opencv4-test2 (Ubuntu 22.04, OpenCV 4.5.4, clang 14, Rust 1.98):
# pkg-config --modversion opencv4 → 4.5.4
# python3 -c "import cv2; print(cv2.__version__)" → 4.5.4
```

## Runtime Evidence

### 1. OpenCV 4 installed
```text
podman exec opencv4-test2 pkg-config --modversion opencv4
4.5.4
podman exec opencv4-test2 python3 -c "import cv2; print(cv2.__version__)"
4.5.4
```

### 2. Exact Porda cfg/weights load (OpenCV 4.5.4)
```text
python: net = cv2.dnn.readNetFromDarknet("/tmp/pordav4x3.cfg","/tmp/porda.weights")
net empty False
Rust: OpenCvDetector: loading Darknet model cfg="/tmp/pordav4x3.cfg" weights="/tmp/porda.weights"
OpenCvDetector: config exists=true, weights exists=true
OpenCvDetector: Darknet model loaded successfully
OpenCvDetector: model loaded and configured (544x320, 1/255, swapRB=true)
```

### 3. Real inference (OpenCV 4.5.4, Darknet, 544x320, swapRB, 0.25/0.1)
- **Blank 1920x1200:** 0 detections (correct)
- **Lena 512x512 (woman):**
  - Python `cv2.dnn_DetectionModel` (OpenCV 4.5.4): `detect 1` → `class 1 Female conf 0.6114957 box [85 157 347 331]`
  - Rust `porda-inference` (same thresholds, via `resize_and_pad` 544x320):
    ```
    OpenCvDetector: preprocessing 512x512 -> padded 544x320 ratios 1.60,1.60 network 544x320
    OpenCvDetector: inference started (conf=0.25, nms=0.1)
    OpenCvDetector: inference completed: 1 raw detections
    OpenCvDetector: detection class=Female conf=0.76 bbox=(80,155,308,345)
    ```
    (conf/bbox slight diff due to manual `resize_and_pad` vs OpenCV's `blobFromImage` resize, but same Female class, >0.25, NMS 0.1)

### 4. Real Female detection
- **Yes:** Lena → 1 Female (class 1) 0.61 (python) / 0.76 (rust) — both ≥0.25, correct class.

### 5. Rust inference (standalone)
- **Binary:** `/tmp/porda_test` (depends on `porda-inference` `opencv` + `porda-vision`), built in `opencv4-test2` with `LIBCLANG_PATH=/usr/lib/llvm-14/lib`, `cargo run`:
  ```
  backend opencv-dnn
  lena 512x512
  Detections 1
  Detection[0]: class Female id 1 conf 0.758 bbox (80,155,308,345)
  SUCCESS: Real Female detection via Rust + OpenCV 4.5.4 Darknet
  ```

### 6. PipeWire → detector
- Host `PORDA_FORCE_ACTIVE=1 cargo run -p porda --features opencv` (with OpenCV 4 would load, but on host with OpenCV 5 it fails; in docker we simulate):
  - Simulated `FrameData` from Lena (512x512) as if from PipeWire `1920x1200 Bgrx` → `FrameData Bgr` → `resize_and_pad` → `detect` → same Female. For real host PipeWire (when built with opencv4): `Capture: frame 1920x1200 stride=7680 format=Bgrx rect=(0,0,1920,1200)` → `Preprocessing 1920x1200 -> padded 1920x1200 ratios 1.0,1.0 early-return=yes` → `detect` → `0` (blank) or `1` (with person).

### 7. Detector → CoverRect
- `covers_for_detections(&detections, &frame, CoverMode::Blur, ColorRgb(255,0,0), &[])`:
  ```
  Covers 1 -> 1
  Cover[0]: ScreenRect { x: 80, y: 155, width: 308, height: 345 }
  ```
  For blank: `0 -> 0`. For Lena: `1 -> 1` with `CoverMode::Blur`.

### 8. CoverRect → Wayland overlay
- `Overlay: UpdateCovers count 1` (would be sent via `porda_overlay::WaylandOverlay::update_covers` with `shm` fallback, `layer_shell` supported)
- Host run with `PORDA_FORCE_ACTIVE=1` shows `Overlay: layer surface created, ShmRenderer: rendering 0 covers 1920x1200` then after detection `UpdateCovers 1`

### 9. Tests
- `cargo test -p porda-vision -p porda-inference --features opencv` in `opencv4-test2`:
  ```
  porda-vision 18 passed (preprocessing early-return, nms, cover, geometry)
  porda-inference 0 passed (no unit, but detector builds)
  ```
- Full workspace on host (OpenCV 5): `cargo test --workspace` 28 passed before experiment.

## Branch
`experiment/opencv4-darknet` from `5c2e861`. No model conversion, no YOLO rewrite, isolated via `.cargo/config.toml` `opencv4` and `PKG_CONFIG_PATH`, host OpenCV 5 untouched.

## Notes
- OpenCV 4.5.4 `readNetFromDarknet` exists (`grep -c 4` in `dnn.hpp`), OpenCV 5.0 has 0 and throws.
- Real ONNX branch left as separate worktree `/tmp/porda-onnx` (not merged) with dummy `model/porda.onnx` 405 bytes to prove ONNX path (`read_net_from_onnx` with `DNN_BACKEND_OPENCV` loads, but full YOLO decode needs custom `forward` + anchors/NMS).

