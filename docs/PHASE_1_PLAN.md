# Phase 1 Plan: Dataset and Detector Feasibility

Date: 2026-09-01

Status: Approved direction, implementation not started

## 1. Phase objective

Determine whether generic COCO-pretrained vehicle detectors can produce reliable per-space parking occupancy on a small, licensed, fixed-camera dataset using the available local hardware.

Phase 1 must end with one evidence-based decision:

1. Continue with SSDLite as the occupancy detector.
2. Continue with Faster R-CNN as the occupancy detector.
3. Pivot from object detection to a parking-space classifier.
4. Revise the data or occupancy-assignment method before choosing a model.

Phase 1 does not build the complete camera-health system. It establishes the occupancy foundation that later health and abstention logic will protect.

## 2. Questions Phase 1 must answer

### Data questions

- Can we retrieve a useful PKLot subset without downloading the full archive?
- Are the selected source images and annotations covered by a documented license?
- Do the annotations provide stable parking-space geometry and occupancy labels?
- Does the sample include enough occupied and vacant examples across weather and distance?

### Model questions

- Can SSDLite detect parked vehicles from a high, oblique camera angle?
- Does Faster R-CNN materially improve detection of distant or occluded vehicles?
- Can generic vehicle boxes be converted into accurate per-space occupancy?
- Is model confidence useful for identifying uncertain occupancy?

### Hardware questions

- Can both models run reliably on the M1 MacBook Air with 8 GB unified memory?
- What are CPU median latency, p95 latency, and peak process memory?
- Does MPS run correctly and provide a worthwhile speed improvement?
- Is any measured limitation serious enough to justify Google Colab?

### Architecture question

- Is generic object detection good enough, or is a parking-specific crop classifier the better foundation?

## 3. Explicit scope

### Included

- PKLot-origin UFPR04 images only
- Up to 300 images
- Sunny, cloudy, and rainy conditions
- Twenty parking spaces across near, middle, and far regions
- Two official Torchvision pretrained detectors
- Two explainable box-to-space assignment rules
- Per-space occupancy metrics
- CPU and optional MPS benchmarking
- Small sensitivity probes for blur, darkness, and camera shift
- A final detector, classifier-pivot, or revise-data decision

### Excluded

- Full PKLot or MetaPKLot download
- CNRPark-EXT or PLds data
- Model training or fine-tuning
- A third detector
- Tracking and dwell-time implementation
- Frozen-stream monitoring
- Full camera-health scoring
- API, dashboard, or cloud deployment
- License-plate recognition
- ONNX work
- Google Colab purchase before a measured local limitation

## 4. Dataset selection

### Source

- Original dataset: PKLot
- Camera subset: UFPR04
- Selective source: PKLot-origin files exposed by the MetaPKLot dataset repository
- Original data license: CC BY 4.0
- Required attribution: original PKLot paper and dataset authors

### Why UFPR04

UFPR04 is one of the three PKLot views and has source annotations, chronological filenames, and weather-organized images. Using one camera isolates model and geometry behavior before adding cross-camera variation.

Choosing one camera in Phase 1 is deliberate. Cross-camera generalization is a later experiment after the basic occupancy pipeline is proven.

### Maximum sample

- Maximum images: 300
- Maximum intended transfer: 500 MB
- Maximum selected spaces: 20
- Weather target: up to 100 images per weather category

The sample may contain fewer than 300 images if the required balance is reached earlier.

## 5. Dataset construction

### Stage A: Metadata discovery

Retrieve directory and annotation metadata without downloading image payloads.

Produce an inventory containing:

- Original relative path
- Camera subset
- Weather category
- Capture date
- Capture time
- Source URL
- Annotation source
- Available occupancy counts

Stop if the selected files cannot be traced to PKLot-origin data and its CC BY 4.0 license.

### Stage B: Deterministic sampling

Use a fixed seed recorded in configuration.

Create three non-overlapping splits:

1. **Smoke split:** 12 images, four per weather category
2. **Calibration split:** 60 images, twenty per weather category
3. **Evaluation split:** up to 228 images, balanced as closely as possible by weather and occupancy

The smoke split validates parsing and model execution.

The calibration split is used to select:

- Detection-confidence thresholds
- Occupancy-assignment method
- Occupancy-overlap threshold
- Any initial `unknown` confidence boundary

The evaluation split remains untouched until configuration is frozen.

Do not tune model or occupancy thresholds on the evaluation split.

### Stage C: Space selection

Select twenty spaces that appear consistently in the UFPR04 annotations.

Divide spaces into three distance proxies using parking-polygon pixel area:

- Near: largest polygon-area third
- Middle: middle polygon-area third
- Far: smallest polygon-area third

Pixel area is only a perspective proxy. It must not be described as physical distance.

Selection must also avoid spaces that are permanently unobservable or missing annotations across many chosen frames.

### Stage D: Occupancy balance

After parsing labels, measure the occupied and vacant frequency for each selected space.

Requirements:

- Each evaluation space must contain both states when possible.
- The evaluation split should avoid being dominated by vacant states.
- Report the final class balance rather than claiming perfect balance.

If several spaces never change state, replace them before freezing the manifest.

## 6. Dataset manifest and provenance

Create a tracked machine-readable manifest. The image files remain ignored by Git.

Required manifest fields:

```text
dataset
source_subset
relative_path
source_url
annotation_reference
weather
capture_date
capture_time
split
sha256
downloaded_at_utc
license_id
license_url
```

Required provenance documents:

- Dataset source and citation
- License text or canonical license URL
- Sampling seed and algorithm
- Selected camera and spaces
- Download date
- File count and total bytes
- Hash verification result
- Known annotation limitations

The downloader must be idempotent:

- Skip verified existing files.
- Redownload corrupt or incomplete files.
- Reject paths outside `data/raw/`.
- Refuse unlisted URLs.
- Fail clearly on hash or HTTP errors.
- Never commit source images.

## 7. Visual data-quality gate

Before installing or running detectors, inspect at least twelve representative images:

- Four sunny
- Four cloudy
- Four rainy
- A mixture of occupied and vacant spaces
- At least one near, middle, and far selected space in view

Verify:

- Image decodes correctly.
- Parking polygons align with spaces.
- Occupancy labels match visible evidence.
- No unexpected zoom or camera change exists within the selected sequence.
- Image resolution is consistent.
- Privacy-sensitive content is not unnecessarily exposed in outputs.

If annotations are visibly wrong, correct the selection or record exclusions. Do not silently repair source labels.

## 8. Dependency installation

Install only after the data-quality gate passes.

Expected Phase 1 runtime dependencies:

- PyTorch
- Torchvision
- OpenCV headless
- NumPy
- Pillow
- psutil for local process-memory measurement

Add exact resolved versions to the environment record after installation.

Do not add:

- Ultralytics
- YOLOX
- Tracking libraries
- Web frameworks
- Dashboard libraries
- ONNX packages

## 9. Detector adapters

Implement one common detector interface so the rest of the experiment does not depend on model-specific output details.

Conceptual contract:

```python
detect(image) -> list[Detection]
```

Each detection must contain:

- Bounding box in original-image pixel coordinates
- COCO category name
- Confidence score
- Model identifier

Allowed vehicle categories:

- Car
- Motorcycle
- Bus
- Truck

Use category names from the official weight metadata. Do not rely on unexplained hard-coded numeric class IDs.

### Detector A: SSDLite320 MobileNetV3 Large

Purpose:

- Establish the lowest-compute baseline.
- Determine whether 320 x 320 inference retains enough detail for distant parking spaces.

Expected risk:

- Distant vehicles may become too small after resizing.

### Detector B: Faster R-CNN MobileNetV3 Large FPN

Purpose:

- Establish the stronger multi-scale baseline.
- Determine whether additional model capacity and feature-pyramid processing materially improve occupancy.

Expected risk:

- Higher latency and memory may not justify a small accuracy improvement.

## 10. Detector smoke test

Run both detectors on the twelve-image smoke split.

For each model, verify:

- Official pretrained weights load successfully.
- Every image returns a valid result without crashing.
- Bounding boxes map back to the original image dimensions.
- Only selected vehicle categories are retained.
- Confidence scores are finite and within the expected range.
- Annotated preview images render correctly.
- CPU execution completes within practical time.

Save preview outputs outside Git unless a small, licensed example is intentionally selected for documentation with attribution.

Stop and diagnose before continuing if model output coordinates or preprocessing are inconsistent.

## 11. Parking-space occupancy assignment

For each image, use the source parking polygons as ground-truth space geometry.

Test two assignment methods.

### Method A: Detection-center rule

A space becomes occupied when the center of an accepted vehicle box lies inside its polygon.

Advantages:

- Simple
- Deterministic
- Easy to explain

Risks:

- A detection center can fall outside an oblique or partially visible space.
- One large detection may cover multiple spaces while assigning to only one.

### Method B: Intersection-over-space rule

Compute:

```text
intersection area between detection box and parking polygon
-----------------------------------------------------------
parking polygon area
```

A space becomes occupied when the ratio exceeds a threshold selected on the calibration split.

Advantages:

- Measures how much of the parking space is covered.
- More suitable than standard IoU when a vehicle box extends beyond a narrow space polygon.

Risks:

- Adjacent-space overlap can create false occupancy.
- Perspective changes the appropriate threshold by distance.

Do not add learned assignment or complicated exception rules in Phase 1.

## 12. Calibration protocol

Use only the 60-image calibration split.

For each model, test a bounded threshold grid:

- Detection confidence: `0.25`, `0.50`, `0.75`
- Intersection-over-space: `0.20`, `0.35`, `0.50`
- Assignment method: center or intersection-over-space

Select one configuration per detector using:

1. Highest occupied-space recall among configurations with acceptable precision
2. Macro F1 as the tie-breaker
3. Simpler assignment rule as the final tie-breaker

Record every tested configuration. Do not manually select thresholds by looking at evaluation results.

## 13. Clean evaluation

Freeze both detector configurations before opening the evaluation split.

For each image-space pair, compare predicted occupancy with the source occupied or vacant label.

Required metrics:

- Macro F1 across occupied and vacant classes
- Occupied-space precision
- Occupied-space recall
- Vacant-space precision
- Vacant-space recall
- Confusion matrix
- Number and percentage of unassigned vehicle detections

Required slices:

- Overall
- Sunny, cloudy, and rainy
- Near, middle, and far polygon-area groups
- Per selected parking space

Do not report only accuracy. A vacancy-dominated sample can make accuracy misleading.

## 14. Minimal degradation sensitivity probe

This is not the full camera-health implementation. It only determines how occupancy responds to degraded evidence.

Select thirty evaluation images without changing their clean labels.

Generate separate degraded copies using:

- Gaussian blur at three severity levels
- Brightness reduction at three severity levels
- Horizontal translation at three pixel offsets
- Small rotation at two angles

For every transformation, record:

- Detector confidence change
- Vehicle detection count change
- Occupancy prediction change
- Macro F1 change
- Newly incorrect confident outputs

Do not use degraded copies to retrain or retune in Phase 1.

The results will determine which camera-health signals and abstention rules Phase 3 should prioritize.

## 15. Runtime benchmarking

### CPU benchmark

CPU benchmarking is mandatory.

Protocol:

- Batch size one
- Five untimed warm-up images
- Time the evaluation split
- Synchronize appropriately around any accelerator timing
- Record preprocessing, inference, and postprocessing separately when practical
- Report median and p95 end-to-end latency
- Record peak process memory when practical

### MPS benchmark

MPS is optional.

Attempt it only after CPU correctness is established.

Requirements:

- Model runs without unsupported-operation failures.
- Predictions remain logically consistent with CPU results.
- Timing includes proper MPS synchronization.
- MPS memory pressure does not destabilize the machine.

If MPS requires silent CPU fallback, report that and keep CPU as the official baseline.

## 16. Google Colab decision gate

Do not buy Colab Pro during the initial pretrained inference experiment.

Colab becomes relevant only if one of these measured conditions occurs:

- Both local runtimes fail due to memory limitations.
- A required classifier pivot needs training that is impractical locally.
- Repeated controlled training runs would take an unreasonable amount of local time.
- A CUDA comparison answers a specific unresolved model question.

Escalation sequence:

1. Confirm the local bottleneck with recorded evidence.
2. Estimate the required memory and compute.
3. Test the current free Colab environment if suitable.
4. Verify current Colab Pro pricing and resource rules.
5. Purchase Pro only when it solves the named bottleneck.

Colab notebooks must call repository scripts and use versioned configuration. The notebook must not become the only implementation.

## 17. Model-selection rules

### Select SSDLite when

- Macro F1 is at least `0.85`.
- Occupied-space recall is at least `0.90`.
- Its Macro F1 is within `0.02` of Faster R-CNN.
- It has materially lower latency or memory use.

### Select Faster R-CNN when

- It meets both accuracy gates.
- SSDLite fails a gate or trails by more than `0.02` Macro F1.
- Its local latency remains practical for the intended sampling cadence.

### Pivot to a parking-space classifier when

- Neither detector reaches `0.85` Macro F1 and `0.90` occupied recall.
- Misses are concentrated in distant or occluded spaces.
- Detector boxes are correct but generic box-to-space assignment remains unstable.

The first classifier candidate will be MobileNetV3 fine-tuned on parking-space crops. Training is a separate approved task and may trigger the Colab decision gate.

### Revise the experiment before model selection when

- Source annotations are unreliable.
- The selected spaces do not contain enough state changes.
- Errors come mainly from incorrect polygon parsing or coordinate transformations.
- Sampling created severe class imbalance.

The thresholds are project decision criteria, not claims about SpotGenius requirements.

## 18. Required implementation structure

Expected additions:

```text
configs/
    phase1.yaml
data/
    manifests/
        pklot_ufpr04_phase1.csv
docs/
    DATASET_PROVENANCE.md
    PHASE_1_RESULTS.md
src/parking_vision_reliability/
    data/
        manifest.py
        download.py
        pklot.py
    detection/
        base.py
        torchvision.py
    occupancy/
        geometry.py
        assignment.py
    evaluation/
        metrics.py
        benchmark.py
    degradation/
        transforms.py
tests/
    data/
    detection/
    occupancy/
    evaluation/
    degradation/
```

This is a target structure, not permission to create empty abstractions. Add modules only when the corresponding implementation begins.

## 19. Test requirements

### Unit tests

- Manifest schema validation
- Safe destination-path enforcement
- SHA-256 verification
- PKLot annotation parsing
- Point-in-polygon behavior
- Intersection-over-space behavior
- Confusion-matrix and metric calculations
- Deterministic degradation transforms

### Integration tests

- Download one explicitly approved small sample
- Parse its annotations
- Run one detector
- Produce valid per-space predictions
- Generate an evaluation record

### Regression fixtures

Use tiny synthetic images and metadata committed to `tests/fixtures/`. Do not commit source dataset images as test fixtures unless their attribution and redistribution are explicitly handled.

## 20. Phase deliverables

Phase 1 must produce:

- Selective dataset downloader
- Versioned dataset manifest
- Dataset provenance document
- Validated PKLot parser
- Two detector adapters behind one interface
- Two occupancy-assignment baselines
- Calibration results
- Held-out evaluation results
- Weather and distance-proxy breakdowns
- CPU and optional MPS benchmarks
- Minimal degradation sensitivity report
- Visual failure-case gallery
- Final model or classifier-pivot decision
- Exact reproduction commands

## 21. Completion gates

Phase 1 is complete only when all applicable gates pass.

### Data gate

- License, citation, source URLs, hashes, file count, and byte count recorded
- Selected images visually checked
- No full dataset downloaded
- No dataset images tracked by Git

### Reproducibility gate

- Fixed sampling seed
- Frozen manifest
- Calibration and evaluation split separation
- Exact package versions recorded
- Commands documented

### Correctness gate

- Source polygons align with images
- Model boxes use original-image coordinates
- Occupancy metrics verified with unit tests
- Both detectors evaluated under the same protocol

### Evidence gate

- Overall and sliced metrics reported
- Latency and memory reported with hardware and protocol
- Failure cases shown, not hidden
- No unsupported production or SpotGenius claims

### Decision gate

- One of the four Phase 1 outcomes is selected with evidence
- Colab is used only if a measured bottleneck justifies it
- No ONNX work is introduced

## 22. Work sequence

Execute Phase 1 in this order:

1. Build metadata inventory and provenance record.
2. Implement deterministic sampling and manifest generation.
3. Download and verify the twelve-image smoke split.
4. Perform visual annotation QA.
5. Install the minimal CV dependencies.
6. Implement the shared detector interface.
7. Run both detectors on the smoke split.
8. Download and verify the calibration split.
9. Implement both occupancy-assignment methods.
10. Select thresholds using calibration only.
11. Freeze configurations and the evaluation manifest.
12. Download and evaluate the held-out split.
13. Run the minimal degradation sensitivity probe.
14. Benchmark CPU and optionally MPS.
15. Produce the result report and model decision.

## 23. Stop conditions

Stop and report before continuing if:

- PKLot-origin licensing cannot be verified for a selected file.
- Selective retrieval requires downloading the full multi-gigabyte repository.
- Source annotations do not align with retrieved images.
- The chosen sample cannot support both occupied and vacant evaluation.
- Either detector requires an unexpected restrictive dependency.
- Local hardware becomes unstable or repeatedly runs out of memory.
- The experiment would require paid Colab before the local baseline is measured.

## 24. Estimated effort

Expected focused effort for a complete, verified Phase 1:

- Data manifest, downloader, and QA: 1 to 2 working days
- Detector adapters and smoke tests: 1 working day
- Occupancy mapping and calibration: 1 to 2 working days
- Evaluation, degradation probe, and benchmarking: 1 to 2 working days
- Results, failure analysis, and decision: 1 working day

Total: approximately 5 to 8 focused working days, depending on source retrieval and detector compatibility.

This estimate excludes a classifier-training pivot, which would become a separately planned extension after the Phase 1 detector results.
