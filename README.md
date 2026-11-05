# D-Sight reconstruction

Camera RGB → U-Net → two surface-gradient channels → differentiable PyTorch
Poisson solver → indentation depth in millimetres. The reconstructed height is
the primary training signal. Both markerless and screen-printed-marker sensors
use this pipeline; marker tracking is outside its scope.

## Setup

Use Python 3.10+ with Tk, OpenCV and a CUDA-enabled PyTorch installation. Install
the remaining dependencies with `python -m pip install -r requirements.txt`.
Choose the CUDA wheel using the [official PyTorch installer](https://pytorch.org/get-started/locally/).
The local `GelSight-homemade` conda environment already had all dependencies when
this implementation was checked, including PyTorch 2.12.1+cu130 and PyVista.

Open scripts in your IDE and run them directly, or use `python script_name.py`.
There are no command-line options. Every workflow script first opens a sensor
selector. Paths resolve relative to this project, independent of the working
directory. Settings live in one `sensors/<sensor>.jsonc` file per physical sensor;
the scripts preserve comments when updating existing values.

## Workflow

1. **`camera_setup.py` — create a sensor and adjust its camera.** Choose markerless
   or marker, set the camera index/backend/resolution, then adjust exposure, gain,
   brightness, contrast and the other driver controls. Apply settings and inspect
   the readback; unsupported controls are reported. Drag the yellow rectangle's
   corners to crop the active elastomer. Save settings and crop, remove all
   contact, then save the unloaded background. DSHOW is the Windows default;
   backend and exposure units depend on the camera driver.
2. **Set physical calibration values in the sensor JSONC.** Enter the measured
   `calibration.ball_diameter_mm` and `calibration.mm_per_pixel`. The supplied
   7 mm and 34/1460 mm/px values are examples inherited from the earlier code.
   Measure pixel pitch using a known distance in the native camera image before
   resizing. Cropping changes the origin, not pixel pitch. Keep camera resolution,
   optical geometry, crop and illumination fixed through collection and inference.
3. **`record_calibration.py` — collect annotation images.** Select the approximately
   100-image mode. Press Space to capture each ball press, or T to toggle timed
   capture. Spread contacts across the active area and indentation range. Q/Esc
   ends capture early. Add a few unloaded frames as negative detector examples.
4. **`annotate_ball.py` — label the circular contact footprint.** Drag a square
   bounding box around the visible contact circle. Drag a corner to resize or
   drag inside to move it. Space saves and advances; S saves; N/P navigate
   without saving; D clears; Q quits. Saving a cleared box explicitly marks a
   no-contact image. Labels use class 0 and normalized YOLO coordinates. Annotate
   the footprint on the elastomer, rather than the outline of the whole ball.
5. **`train_detector.py` — train the contact detector.** It creates a reproducible
   train/validation split from saved annotations, trains one-class YOLO, and writes
   the best weights path to `paths.yolo_checkpoint`. The default pretrained
   `yolo11n.pt` may download on first use. Set `yolo.model` to a local weights file
   to use an existing model. Color and flip augmentation are disabled to preserve
   the directional illumination pattern. With `yolo.image_size: null`, native
   crop pixels are retained and only bottom/right stride padding is added; labels
   are adjusted to the padded canvas. Geometric resize/scale augmentation is off.
6. **`record_calibration.py` — collect reconstruction images.** Choose the
   approximately 2000-image automatic mode. It loads the selected sensor's trained
   `paths.yolo_checkpoint` and immediately starts watching for the ball. Press and
   lift the ball at diverse positions and depths. Each saved image must pass the
   contact checks described below, including during lift-off. Capture saves the
   exact frame YOLO checked, rather than later frames from the preview. No start
   key is needed. P pauses/resumes, and Q/Esc exits. The
   overlay shows waiting/contact/release/paused state and the detected box.
   Files are cropped native-resolution PNGs. Capture appends filenames and stops
   after `record_count` new **accepted** images for this session. The writer also
   saves their contact geometry and capture settings in `recording/contacts.json`.
   Failed/invalid detections do not create PNGs or increment the count. Annotation mode remains
   manual/timed, so it can be used before training a detector.
7. **`build_dataset.py` — generate supervised depths.** It reads `contacts.json`
   and converts every accepted capture to an `H × W × 1` float32 target in mm.
   It uses the saved geometry without rerunning YOLO or rejecting images again.
   Inspect the saved overlays in
   `data/<sensor>/dataset/preview` before training: detection errors directly
   become supervision errors. `manifest.json` lists all converted captures.
   Older recordings without `contacts.json` remain untouched; record a new
   accepted set with the updated recorder before rebuilding. Existing generated
   datasets already contain native-resolution RGB/depth and can be used directly
   to retrain the U-Net at native resolution.
8. **`train.py` — train the reconstruction U-Net.** With `training.image_size: null`
   it uses the dataset's original cropped resolution without resizing. It caches
   the full dataset when it fits the configured RAM
   budget, and otherwise streams from disk. Best validation weights are saved
   to `data/<sensor>/checkpoints/best_neumann.pt` and their path is written to the
   sensor config. `history_neumann.csv` records losses and epoch durations. Legacy
   Dirichlet runs retain `best.pt` and `history.csv`. A new run replaces the
   previous best checkpoint and history for the same boundary mode.
9. **`demo.py` — live reconstruction.** It restores the best model and the saved
   training preprocessing, opens the camera, and shows RGB alongside a fixed-scale
   depth heatmap. Press Q/Esc to exit. Enable `demo.pyvista` in the config, or set
   `SHOW_POINT_CLOUD` at the top of the script, to display a PyVista point cloud.
   Coordinates and depths are in millimetres; `point_stride` controls its density.

`infer_video.py` runs the same reconstruction on a video. Set `VIDEO_PATH` at its
top to a **full, uncropped camera recording**; the checkpoint crop is applied.
`circle_detection.py` is an alias for `build_dataset.py`.

## Live display performance

Camera capture runs continuously on a separate thread. Each live view consumes
the newest frame, so a slow render skips obsolete preview frames instead of
building up latency. Camera setup polls without blocking Tk, reuses a Pillow
image buffer, and updates crop handles in place. Annotation caches the scaled
image and redraws only when the box or controls change.

The `fourcc` camera field requests `MJPG` by default, including for older configs
without that field. It can reduce USB bandwidth at HD resolutions when supported
by the camera. Use the camera setup's fourcc entry to select another supported
format, or leave it empty for driver default. Capture resolution and saved PNG
dimensions remain unchanged. Actual support depends on the backend and driver;
see [OpenCV camera properties](https://docs.opencv.org/4.12.0/d4/d15/group__videoio__flags__base.html).
Save a new unloaded background after changing camera format or settings.

Camera setup, recording and the live demo display measured **Camera** and
**Display** FPS separately. A low camera rate points to capture mode, exposure,
USB bandwidth or the driver. A high camera rate with low display FPS points to
rendering or reconstruction. The requested FPS setting is not a measurement.

Recording previews are reduced to fit 960×720, while captured images retain the
full crop resolution. PNG writing runs in a worker with at most four outstanding
images. If the disk cannot keep up, `Writer busy` is shown and new captures wait
for room; accepted saves finish before exit. The optional PyVista cloud refreshes
at `demo.cloud_fps` (default 15) while the 2D view continues at its own rate.

Automatic recording uses a separate YOLO worker with only one inference in
flight, so slow detection never builds up a queue of old frames. The calibration
settings are `detection_fps` (default 10 checks/s), `capture_interval_s` (default
0.25 s between saved images) and `release_delay_s` (default 0.6 s of UI contact
hold after the last valid detection). The hold never admits unchecked or unloaded
frames. A saved sample always has its own successful detector/geometry result.
Existing configs without the new fields use those defaults. Detection uses the
sensor's `yolo.confidence` and `yolo.image_size`. Detection and saving may be slower
at native resolution; the camera preview remains independent. The target now
means 2000 accepted samples, not 2000 candidates that will be filtered later.

## Native resolution and capture acceptance

Both `yolo.image_size` and `training.image_size` default to `null`, meaning the
original crop. For the current `big` sensor this is **1292×1080** (width×height).
U-Net and Poisson reconstruction use that exact size. YOLO's stride of 32 requires
**1312×1088** after padding; the original 1292×1080 pixels are never resampled.
YOLO validation can add a further padding rim for batch alignment. Training
uses rectangular batches and disables geometric resizing augmentations.

Default batch sizes are 2 for YOLO and 1 for U-Net to accommodate the larger images.
The native RAM cache for 2000 images at 1292×1080 is approximately 18.2 GiB, below
the 32 GiB cap. Explicit old input sizes are still supported for earlier workflows.
Previously trained U-Net checkpoints keep their stored input resolution in the
demo; retrain with the new native settings to obtain native-resolution inference.
Existing weights and recorded images are not converted or deleted automatically.

The automatic recorder accepts the highest-confidence valid class 0 contact:

- YOLO confidence meets the sensor's configured threshold.
- Box width/height are positive, with larger/smaller side ratio at most 1.3.
- The box and inferred circle are fully inside the crop with an unloaded rim.
  Native-resolution training uses a 2-pixel rim at source resolution; an explicit smaller
  training size uses a larger margin to protect the rim after downsampling.
- Contact radius `(box_width + box_height) / 4 × mm_per_pixel` is smaller than
  the physical ball radius.

Rejected detections are shown during capture and are not saved. Contact metadata
is written only after its PNG is saved successfully. Repeated captures append to
the same index when camera/calibration settings match; use a new recording folder
after changing those settings. Dataset generation trusts this accepted index.

On the tested machine, 1920×1080-to-960×540 Tk preview conversion/rendering fell
from approximately 45 ms per frame with PNG/base64 to 3.5 ms with the reusable
image buffer. The old additional 30 ms UI delay was removed. These are synthetic
display measurements; actual live FPS still depends on camera capture.

## Geometry and training conventions

For physical ball radius `R`, detected contact radius `a`, and distance `r` from
the contact centre, all in millimetres:

```text
height(r) = sqrt(R² - r²) - sqrt(R² - a²),  r <= a
height(r) = 0,                            r > a
```

Depth is positive indentation. A bounding box gives the contact centre and
radius `(box_width + box_height) / 4` in source pixels. The known ball diameter
alone does not determine pixel pitch. This model assumes circular contact and
approximately square, constant-size pixels over the active region.

Network tensors are `B × 3 × H × W` RGB, normalized to `[0,1]` and optionally
background-subtracted. Output is `B × 2 × H × W`: forward pixel differences
`[dz/dx, dz/dy]` measured as mm per network-pixel step. Resizing height maps does
not rescale their millimetre values. X/Y physical point spacing is adjusted
independently after resizing, so a changed image aspect ratio retains the source
physical dimensions in the point cloud.

The solver minimizes the discrepancy between the surface's forward pixel
differences and the two predicted gradient channels. It supports two boundary
modes, both batched and differentiable on CUDA:

- **`reconstruction.boundary: "neumann"` (new default):** integrate the full grid,
  including boundary gradient flux, with a DCT-based Poisson solve. The edge
  heights are free to vary. This is the natural gradient-matching boundary
  condition, **not** an assumption of zero normal slope.
- **`"dirichlet"` (legacy):** use the DST-I solver and set every outer edge pixel's
  height to zero. This is appropriate only when that entire perimeter is known
  to remain undeformed.

Neumann integration determines shape up to a single additive height offset.
`reconstruction.reference_region: null` subtracts only the **mean perimeter
height**. Individual perimeter pixels remain free to tilt or deform. If that
perimeter is loaded, the result is relative height rather than an absolute
indentation measurement. For a physical zero reference, set this field to a
known unloaded patch `[left, top, width, height]` in original cropped-image pixels.
Only its mean is anchored; the patch is not flattened. The trainer maps the patch
to its input resolution and stores the mapped coordinates in the checkpoint.

Training aligns the target height with the same reference before computing depth
MSE. `training.gradient_loss_weight` remains unchanged and defaults to zero;
training and validation use the same loss weights. Calibration still requires a
complete interior ball footprint so its geometry can be estimated correctly.
New Neumann training runs initialize the gradient head to zero, representing an
initially flat surface instead of an arbitrary large tilt.

The demo restores the **checkpoint's** boundary and reference, keeping training
and inference consistent. Checkpoints without this metadata use the earlier
Dirichlet solver and display a retraining notice if the sensor config now requests
Neumann. Run `train.py` with the new setting to create `best_neumann.pt`; existing
datasets can be reused and earlier checkpoints remain intact. Changing the
config alone does not reinterpret an old model's gradients.

Relative Neumann heights can be negative. The heatmap and PyVista colors default
to `[-max_depth_mm, +max_depth_mm]` for these checkpoints; legacy Dirichlet colors
start at zero. Set `demo.min_depth_mm` explicitly to choose another display range.
Color limits affect visualization only, never the reconstructed height tensor.

The U-Net can use CUDA AMP; integration and losses stay float32. The RAM cache
uses uint8 RGB plus float32 heights (7 bytes per training pixel per sample), shared
by train/validation subsets. The budget is capped at 32 GiB and reserves space
for decoded frames and batches. Pinned batches transfer to CUDA asynchronously;
normalization and background subtraction occur on the GPU. The streaming worker
count is configurable; cached data uses no worker processes to avoid duplicating
it on Windows. GPU throughput depends on image size, batch size and hardware.

Checkpoints embed the training-resolution unloaded background, model width, input size,
source pixel pitch and camera settings/crop. The demo replays those camera
settings using the selected sensor's current camera index. Recollect/retrain
after changing camera geometry or illumination. Checkpoints from the old periodic
solver require retraining: the old integration sign, boundary convention and
preprocessing metadata differ.

Marker and markerless configs have independent settings. Initial marker defaults
use 32 base channels and detector confidence 0.6; markerless uses 16 and 0.5.
These are editable starting points, not experimentally optimized presets.
Random validation splits are reproducible but neighbouring recorded frames can
be correlated; assess final accuracy on separate contacts and recording sessions.

## Verification

Run `python -m unittest discover -s tests -v` in this folder. Tests exercise
numerical reconstruction/sign/boundaries, solver gradients, CPU and CUDA AMP
training, cached versus streamed samples, checkpoint reuse, calibration geometry,
annotation interactions and comment-preserving config updates. Synthetic tests
do not measure real camera controls, detector accuracy or physical reconstruction
quality; those require a sensor and collected calibration data.

API references: [PyTorch FFT](https://docs.pytorch.org/docs/stable/fft.html),
[Ultralytics training](https://docs.ultralytics.com/modes/train/),
[Ultralytics prediction](https://docs.ultralytics.com/modes/predict/),
[PyVista live updates](https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.update).
