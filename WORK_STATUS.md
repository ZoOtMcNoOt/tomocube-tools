# Toolset extension

Base: `33eb669` (merged PR #2), fetched and verified clean on 2026-09-12.
Branch: `improve/analysis-registration`.

## Implementation complete

- Header-only inventory, calibration provenance, selective channel reads and bounded native ROI/batch/timepoint statistics.
- One argparse CLI, PNG sequences, anisotropic profiles, consistent timepoint/channel options and explicit invalid-option errors.
- One calibrated affine for 2D, native 3D and exports; conservative image-based residual translation with inspectable, acquisition-bound JSON reports and software provenance.
- Registered TIFF/MAT/display export, atomic single-file publication, complete PNG sequence publication, original-data preservation and malformed-projection rejection.
- Current napari geometry/Qt APIs, native clipping, supported-slice guards, animation state/timing/cleanup corrections and reproducible synthetic example.
- Removed obsolete standalone registration math, unused path configuration and unused MATLAB extras. Documentation covers behavior/API changes and scientific limits.

## Review and evidence

Independent review found and verified fixes for translation-invariant/periodic false acceptance, missing external FL search support, malformed reports, partial PNG publication, MAT channel mislabeling/projection validation and unnecessary 3D channel loading. Final nine prior alignment reproductions pass; no unresolved defect within those reviews.

- Full suite: 349 passed / 5 optional skips on local Python 3.10 and 3.14 and installed Python 3.12 wheel; final report-provenance addition separately passed all 34 alignment/workflow checks.
- Napari 0.9.1 and real PyQt6 controls: 44 tests passed, including native geometry, dock callbacks, QTimer cancellation and cleanup.
- Source and wheel builds passed. Final wheel verification and remote CI are in progress before PR closeout.
- Synthetic example records known-transform errors and produces registered TIFF, analysis/inventory JSON and a rendered 2D viewer under `output/toolset-qa`; source recipe is `examples/registration_workflow.py`.

## Limits and remaining closeout

- No experimental TCF acquisitions were available. Positive-intensity translation matching does not establish biological correspondence; no experimental accuracy claim is made.
- Windows offscreen Qt cannot create napari's OpenGL context (error 1282). Real Qt widgets passed; full GPU volume rendering remains unverified. Unsupported native rotated slice directions are guarded.
- Analysis uses native grids and bounded Z blocks, not segmentation or a fixed byte budget. Some exporters materialize one selected volume.
- Remaining: final installed-wheel checks, push/open PR, remote CI results and concise final handoff. No permission to merge a new PR is assumed.
