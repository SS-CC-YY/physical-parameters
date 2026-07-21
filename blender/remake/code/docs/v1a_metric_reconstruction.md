# V1A metric object-trajectory reconstruction

## Scope and interpretation

This module reconstructs the standard sphere trajectory for experiment V1A from one generated MP4 at a time. Main, Side and Top videos are independent model generations, so they are **not** frame-synchronised views and are never triangulated together.

V1A defines a known world motion line

\[
X(z) = (x_0,y_0,z)^\top,\qquad x_0=y_0=0,
\]

therefore a calibrated single camera can recover metric \((x_0,y_0,z_t)\) for every detected sphere centre. This is experiment-constrained metric 3D, not a reconstruction of the complete scene. The prompted gravity value is withheld until trajectory reconstruction and fitting have finished.

## Processing order

1. `audit_camera_motion.py` first classifies the whole video. Only a positive `fixed` / `no_significant_camera_change` result permits the static calibrated route. Changed, borderline, missing and failed audits are routed to SpatialTrackerV2 by `rebuild-test/spatialtrackerv2/scripts/run_hybrid_v1a.py`.
2. `export_v1a_calibration.py` opens each frozen V1A `.blend` read-only and exports the 1280×720 camera intrinsics, OpenCV world-to-camera extrinsics, object radius, motion line, support plane, source hash and static-background world anchors. The 9 scenes × 3 cameras are bundled under `code/assets/v1a_calibration`, so ordinary reconstruction does not require Blender on the server.
3. The frozen conditioning PNG is decoded and centre-crop/resized to the generated resolution. Static background features (with the complete object-motion corridor masked) register that image to decoded video frame 0. Registration failure makes metric reconstruction invalid. For an audited-static video this registration is applied once and the resulting camera projection remains fixed for every frame.
4. On the dynamic calibrated path, static Blender world anchors are observed locally by forward/backward LK from the registered conditioning image into decoded frame 0. The real per-anchor video pixels, not homography-generated theoretical pixels, initialize `solvePnPRansac`; subsequent frames track those anchors from frame 0. Frame 0 and every primary-interval frame must pass RANSAC-majority, all-correspondence residual and LM-refinement gates. Failed PnP frames remain in the interval accounting rather than disappearing from the denominator.
5. The orange sphere is detected only inside the calibrated projected motion corridor. Colour components, a local motion prediction and a Hough-circle fallback are combined. Hard gates reject large merged blobs, non-circular shapes, large radius changes and distant indoor props.
6. On the audited-static route, the observed circle radius must be within `0.65–1.35` of the radius obtained by projecting the known Blender sphere at the recovered metric centre. A violating frame is excluded from the metric trajectory and physics fit. The box drawn for visual QA has a locked size and never changes the physical measurement.
7. Pixel centres are lifted to metric height. A robust quadratic is fit to the primary airborne trajectory, and only then is the estimated gravity compared with the target.

## Geometry

For frame \(t\), the calibrated projection is

\[
P_t = K_t [R_t\mid t_t].
\]

Writing

\[
P_t[x_0,y_0,z,1]^\top = a z+b,
\]

an observed sphere centre \((u,v)\) supplies two linear equations in the single unknown \(z\):

\[
\begin{bmatrix}
u a_3-a_1\\
v a_3-a_2
\end{bmatrix}z=
\begin{bmatrix}
b_1-u b_3\\
b_2-v b_3
\end{bmatrix}.
\]

The implementation uses the least-squares solution and reports the pixel reprojection residual, local pixels-per-metre observability and a first-order heuristic height uncertainty propagated from centre, frame-0 registration and PnP residuals. It does not claim a full covariance estimate for K/R/t. The line residual is a geometric audit but is not claimed as independent identity evidence because the same motion corridor participates in candidate filtering. Apparent-radius consistency provides a separate rigid-sphere warning.

The physics fit is

\[
z(t)=z_0+v_0(t-t_0)-\frac{1}{2}g(t-t_0)^2.
\]

It uses robust Huber reweighting. The target \(g\) is not an input to detection, camera pose, metric lifting, release/contact selection or the quadratic fit.

## Reproduction

Export all 9 scenes × 3 cameras, including background geometry anchors:

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background `
  --python code/scripts/export_v1a_calibration.py -- `
  --output-dir rebuild-test/calibration/v1_A `
  --anchor-stride-px 32 --overwrite
```

Run the two hardest indoor scenes for Main and Top:

```powershell
python code/scripts/run_v1a_metric_reconstruction.py `
  --videos rebuild-test/test-videos/cam-test `
  --calibration-root rebuild-test/calibration/v1_A `
  --output rebuild-test/outputs/v1a_metric_indoor34_main_top `
  --glob "v1_A__g9p81__indoor[34]__standard_ball__CAM_*__seed-*.mp4" `
  --overlay-count 4
```

Build and validate against all 12 clean Blender GT videos:

```powershell
python code/scripts/build_v1a_gt_validation_manifest.py `
  --output rebuild-test/validation/v1a_clean_gt_manifest.json

python code/scripts/validate_v1a_metric_against_gt.py `
  --manifest rebuild-test/validation/v1a_clean_gt_manifest.json `
  --calibration-root rebuild-test/calibration/v1_A `
  --output rebuild-test/outputs/v1a_metric_clean_gt_all12
```

## Current validation result

The complete clean-GT set contains 4 gravity values × 3 cameras = 12 videos, 192 frames each. The strict frame-ID validator requires all 192 prediction and GT rows to join exactly; duplicate, missing, extra or sparsely valid rows fail. All 12 pass:

- compared frames: 2304/2304;
- valid-frame fraction: 100% for every job;
- median position error across jobs: 3.95–4.54 mm;
- worst job P95 position error: 5.42 mm;
- worst gravity relative error: 0.521%;
- worst frame-0 calibration check: 0.943 px;
- thresholds: median ≤30 mm, P95 ≤80 mm, gravity error ≤5%.

The six Seedance indoor3/indoor4 Main/Side/Top trials all pass frame-0 registration, primary-frame PnP (100%), background tracking and metric-airborne coverage. Their overlays follow the actual sphere rather than indoor props. Four gravity fits fail the target comparison, one is an invalid free-fall shape, and indoor3/Side is insufficient because its first fall contains only four sampled motion points. All six also raise the separate rigid-sphere scale warning because the generated apparent radius is not consistent with the Blender sphere radius. These are findings about the generated videos, not reasons to force the centre trajectory toward the prompt value. Because generated videos have no hidden camera/trajectory GT, this six-video result establishes strict geometric self-consistency and visual identity correctness, not an independent millimetre-scale error bound under arbitrary generated camera drift.

## Key outputs

Each job directory contains:

- `trajectory_metric.csv`: frame-level centre, measurement radius, metric xyz, uncertainty, line residual and fit membership;
- `camera_pose.jsonl` and `camera_pose.csv`: per-frame pose quality and diagnostics;
- `background_motion.csv`: independent background homography diagnostics;
- `trajectory_metric.png`: metric height-time and 3D route plots;
- `object_track_overlay.mp4`: detected measurement circle, stable QA box and image-plane route;
- `result.json`: validity gates, physics fit, interpretation boundary and artifact paths.

`strict_dynamic_camera_metric_3d_valid=true` requires successful frozen-frame registration, calibrated anchor PnP on the primary/fit frames, bounded PnP failure runs, background success and the complete release-to-contact coverage gates. A projective homography can draw a 2D diagnostic overlay, but its rows have `measurement_valid=false`; it can never pass constrained metric 3D or enter the physics fit.

Clean GT uses a validator-scoped `verified_static_camera` mode because those legacy renders predate the frozen background assets. It is not accepted from a normal experiment config alone: each source `.blend` is opened read-only in Blender, all three evaluated camera matrices and lens/shift values are audited over frames 1–192, and the structured result is bound to the source-blend and audit-script SHA-256 values in the manifest. K/R/t is then checked directly against the observed frame-0 ball centre, and all reconstructed frames are compared with GT. Long-baseline background LK remains a warning-only codec/texture diagnostic in this hash-pinned path; it remains a hard gate for generated videos. This validates metric lifting and fitting for all three static camera geometries; synthetic 6-DoF tests validate the PnP implementation. Generated-video PnP has no hidden 3D ground truth, so it is additionally audited by frame-0 registration, world-anchor residuals and independent background motion.

This module reconstructs the V1A experiment-constrained route `(x0, y0, z)`. It does not claim unconstrained monocular scene reconstruction. A video in which the object truly leaves the declared line requires another experiment-specific constraint, synchronised multi-view observations, or a separately validated monocular-depth model.
