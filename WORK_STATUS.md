# Toolset extension

Base: `33eb669` (merged PR #2), fetched and verified clean on 2026-09-12.
Branch: `improve/analysis-registration`.
Delivery: [PR #3](https://github.com/ZoOtMcNoOt/tomocube-tools/pull/3), unmerged. Runtime implementation: `1eb1fdc`; subsequent changes update tests and this status.

## Implementation complete

- Header-only inventory, calibration provenance, selective channel reads and bounded native ROI/batch/timepoint statistics.
- One argparse CLI, PNG sequences, anisotropic profiles, consistent timepoint/channel options and explicit invalid-option errors.
- One calibrated affine for 2D, native 3D and exports; conservative image-based residual translation with inspectable, acquisition-bound JSON reports and software provenance.
- Registered TIFF/MAT/display export, atomic single-file publication, complete PNG sequence publication, original-data preservation and malformed-projection rejection.
- Current napari geometry/Qt APIs, native clipping, supported-slice guards, animation state/timing/cleanup corrections and reproducible synthetic example.
- Removed obsolete standalone registration math, unused path configuration and unused MATLAB extras. Documentation covers behavior/API changes and scientific limits.

## Review and evidence

Independent review found and verified fixes for translation-invariant/periodic false acceptance, missing external FL search support, malformed reports, partial PNG publication, MAT channel mislabeling/projection validation and unnecessary 3D channel loading. Final nine prior alignment reproductions pass; no unresolved defect within those reviews.

- Full suite: 349 passed / 5 optional skips on local Python 3.10 and 3.14. The final installed Python 3.12 wheel passed 347 tests plus 73 focused checks after two 3D fixtures were updated for the completed report schema; the full rerun is covered by PR CI.
- Napari 0.9.1 and real PyQt6 controls: 44 tests passed, including native geometry, dock callbacks, QTimer cancellation and cleanup.
- Source and wheel builds passed; the wheel's Python sources match the checkout. Exact final-revision CI results are attached to [PR #3 checks](https://github.com/ZoOtMcNoOt/tomocube-tools/pull/3/checks): Linux 3.10/3.14, Windows 3.12, installed-wheel reruns, and a separate Windows napari/PyQt6 job. Encoding tests require the optional imageio dependency and execute in the 3D job; a core-only install skips them.
- The installed-wheel synthetic workflow succeeded in `output/toolset-qa-final`: absolute ZYX translation errors were 0.01006, 0.00214, 0.00042 micrometers for this example. It produced registered TIFF, software/calibration/analysis JSON and a rendered 2D viewer. Recipe: `examples/registration_workflow.py`.

## Remaining validation limits

- No experimental TCF acquisitions were available. Positive-intensity translation matching does not establish biological correspondence; no experimental accuracy claim is made.
- Windows offscreen Qt cannot create napari's OpenGL context (error 1282). Real Qt widgets passed; full GPU volume rendering remains unverified. Unsupported native rotated slice directions are guarded.
- Analysis uses native grids and bounded Z blocks, not segmentation or a fixed byte budget. Some exporters materialize one selected volume.
- No permission to merge a new PR is assumed; the reviewable branch and source data remain intact.
