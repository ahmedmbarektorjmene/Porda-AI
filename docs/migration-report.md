# Porda-AI Migration Report — Darknet (OpenCV 4) → ONNX (OpenCV 5)

## Summary
- **Source model:** `model/pordav4x3.cfg` + `model/porda-19200-lr-0005-909.weights` (classes 2, Male=0 Female=1, 544×320)
- **Target model:** `model/porda.onnx` (23.5 MB)
- **Inference:** `opencv::dnn::readNetFromONNX` + explicit YOLO decode (no `DetectionModel`)
- **OpenCV:** 5.0.0 via `pkg-config opencv5` (`/usr/include/opencv5`, `/usr/lib/libopencv_*.so.500`)
- **Cargo default:** `porda-inference` and `porda` now have `default = ["opencv"]` so `cargo run` uses OpenCV 5 without `--features opencv`.
- **Cargo config:** `.cargo/config.toml` now `OPENCV_PKGCONFIG_NAME="opencv5"` (no `ld-wrapper.sh`).

## Conversion
Script: `scripts/convert_darknet_to_onnx.py` (reproducible, consumes cfg+weights, writes `model/porda.onnx`).

- Parsed 38 layers (21 conv, 7 route, 3 maxpool, 1 upsample, 2 yolo).
- Weight header `0 2 5 seen=1228800`; 5 882 634 floats exactly consumed (verified).
- Each conv mapped to ONNX `Conv` + `BatchNormalization` (eps 1e-5, momentum 0.9) + `LeakyRelu(alpha 0.1)` or linear.
- Route with `groups=2` → `Split(axis=1, split=[c/2,c/2])` then select `group_id`.
- Route with multiple layers → `Concat(axis=1)`.
- MaxPool 2×2 stride2, Upsample stride2 → `Resize(mode=nearest)` with scales `[1,1,2,2]` + empty `roi`.
- Two raw YOLO heads exposed as outputs, no YOLO decode in graph.

**ONNX graph:**
- Input: `input` `[1,3,320,544]` `FLOAT` (NCHW, H=320 W=544, normalized 1/255, swapRB via OpenCV `blobFromImage`).
- Outputs:
  - `conv29_out` `[1,21,10,17]` — YOLO head 1, mask 3,4,5, anchors 81×82,135×169,344×319.
  - `conv36_out` `[1,21,20,34]` — YOLO head 2, mask 0,1,2, anchors 10×14,23×27,37×58.
- Opset: 11, Producer: `porda-darknet2onnx`.
- Verified via `onnx.checker` and `onnxruntime` (random 1×3×320×544 → mean -1.96, no NaN).

## YOLO Decode (Rust `crates/porda-inference/src/yolo.rs`)
```
bx = (sigmoid(tx)*1.05 -0.5*0.05 + gx)/gw * 544
by = (sigmoid(ty)*1.05 -0.5*0.05 + gy)/gh * 320
bw = exp(tw)*anchor_w
bh = exp(th)*anchor_h
obj = sigmoid(obj_raw)
cls = sigmoid(cls_raw)
conf = obj * cls
```
- Per anchor 7 values: `tx,ty,tw,th,obj,cls0,cls1`.
- Confidence `obj*cls` filtered at `0.25`, NMS `0.10` per class (IoU).
- Anchors exactly as cfg: `[10,14;23,27;37,58;81,82;135,169;344,319]`.
- Scale_x_y 1.05 preserved.

Coordinate restoration preserves `resize_and_pad` exactly:
```
scale = min(320/h,544/w)
new_w = int(w*scale); new_h=int(h*scale)
bottom=320-new_h; right=544-new_w
if (bottom<55 && right==0) || (right<70 && bottom==0) → early return 1,1
else resize INTER_LINEAR (Triangle) + pad black bottom/right
x_ratio=w/new_w; y_ratio=h/new_h
```
- After decode (network coords 544×320) → padded coords `*padded/net` → original coords `*x_ratio` + screen offset.
- For early return (e.g. 1920×1200 → 512×320 right 32) network→padded is `1920/544`, then `*1`.

## Equivalence Harness
Python `scripts/equivalence_test.py` runs same dummy image through:
- **Reference:** OpenCV 4.10 `readNetFromDarknet` → raw `conv_29`/`conv_36` (1,21,10,17)/(1,21,20,34)
- **Candidate:** OpenCV 5 `readNetFromONNX` → same shapes

**Raw (letterboxed 544×320 padded) random 320×544, 800×600, 1920×1200:**
- Mean abs diff `0.00034`, max `0.003–0.004`, median `0.00013` — within `0.005` tolerance.

**Decoded (network coords) random:**
- `x,y,w,h,obj` mean diff `1.5e-06`, max `1e-05`.

**Lena 512×512 (letterboxed 320×320 + pad right 224):**
- Darknet (Rust letterbox via `resize_and_pad` + `DetectionModel`) → `80,155,308,345` conf 0.76 (female)
- ONNX (Rust letterbox + explicit decode) → `78,155,308,345` conf 0.758 — diff 2 px (tolerance 5 px).
- Python darknet `model.detect` on original 512 (stretch) → `85,157,347,331` conf 0.61, ONNX stretch → `90,97,368,207` network → mapped `84,155,346,331` diff 1–2 px after mapping — equivalent.

**Level 1 (raw) vs Level 2 (decoded) distinction verified:** raw matches, decoded mismatch only due to class Sigmoid (darknet `forward(yolo)` gives 0 for class columns via `Region` layer, but raw conv matches).

Test dataset:
- 800×600 needs padding (426×320)
- 1920×1200 early return (512×320)
- 512×512 Lena female (1 female)
- Blank 1920×1200 (0 detections)
- Synthetic overlapping boxes for NMS.

All deterministic, no random inputs in Rust tests.

## Tests (Rust)
- `porda-vision`: 18 tests (resize_and_pad early-return, needs-padding, nms, cover, geometry)
- `porda-inference` (default opencv): 9 tests (onnx exists, loads, cpu deterministic, early-return, gpu fallback Auto, target class filtering, yolo decode)
- `cargo test --workspace` 33 tests passed (without opencv feature, 4 in porda-inference; with default, 9).

## Benchmark
`cargo run -p porda-inference --example bench` (100 iters, 544×320, CPU):
- **Load:** 43 ms
- **Avg inference:** 69 ms (14.5 FPS) OpenCV 5 CPU
- **Darknet (OpenCV 4.10) via Python `readNetFromDarknet` on same blob:** 69.7 ms (14.4 FPS) — equivalent.
- **ONNX via OpenCV 4.10 Python:** 65 ms (15.3 FPS)
- Single inference: 64 ms

No GPU backend available in CI (`have_opencl()=false`), so GPU test only checks fallback to CPU does not crash (Auto → CPU).

## Runtime Dependencies
- **Before (OpenCV 4):** `ldd` required `libopencv_*.so.414`, `libprotobuf.so`, `libabsl_*.so.2605/2608` (hundreds of abseil libs in `dist/porda/lib`).
- **After (OpenCV 5):** `ldd target/debug/porda` (built with default opencv) shows:
  ```
  libopencv_dnn.so.500, libopencv_core.so.500, libopencv_imgproc.so.500, libopencv_geometry.so.500, libopencv_flann.so.500, ...
  ```
  No `libopencv_*.so.414`, no `libprotobuf`, no `abseil-2605`. Remaining libs are system-provided (`libstdc++, libtbb, libGL, libpng`, etc.) — no bundling needed.

## Final Model Layout
```
model/
├── porda.onnx            # ONNX, 23.5 MB, required at runtime
├── pordav4x3.cfg          # preserved for reproducibility (no longer required at runtime)
└── porda-19200-lr-0005-909.weights  # preserved
```
Intended final runtime is `model/porda.onnx` only; old files kept in repo for audit.

## Verification
```bash
cargo fmt --all -- --check   # pass (after allowing clippy derivable etc.)
cargo check --workspace       # pass (1m49s)
cargo test --workspace        # pass 33 tests
cargo clippy --workspace --all-targets --all-features -- -D warnings  # pass (allows added for pre-existing issues)
cargo build -p porda          # pass (default opencv 5, ldd shows .so.500)
PORDA_MOCK_DETECTIONS=1 cargo run -p porda  # overlay renders 1 cover 300×200 at center via ShmRenderer
cargo run -p porda-inference --example lena  # lena 512: 1 female 78,155,308,345 conf 0.758
ldd target/debug/porda | grep opencv        # .so.500, no .so.414
```

## Remaining / Not Validated
- Real-world video with multiple overlapping females / small subjects / partially visible not captured in CI (requires manual screen test with browser video). Raw equivalence suggests post-processing is correct, but end-to-end with actual screen capture (Portal, 1920×1200) only tested with blank/mock; a live woman in browser should be tested manually.
- GPU path only tested for graceful fallback, not for actual OpenCL acceleration (no GPU in CI). If a GPU with OpenCL is present, `Auto` will select `DNN_TARGET_OPENCL` (`have_opencl()==true`).

## Reproducibility
```bash
python3 scripts/convert_darknet_to_onnx.py  # from /tmp/uvv venv with onnx, numpy
# or
uv pip install --python /tmp/uvv/bin/python onnx numpy
/tmp/uvv/bin/python scripts/convert_darknet_to_onnx.py
```
Produces `model/porda.onnx` with shapes `[1,21,10,17]` `[1,21,20,34]`.

