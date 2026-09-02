# Dataset and Detector Feasibility Research

Date: 2026-09-01

Status: Research decision complete. No project dataset or detector has been downloaded or installed yet.

## 1. Decision summary

Use a small, explicit subset of **PKLot** for the first feasibility experiment. PKLot is the best initial fit because it provides fixed-camera parking images, parking-space polygons, occupied or vacant labels, three camera views, real weather variation, and a clear CC BY 4.0 license.

Do not download the full 4.6 GB archive. Pull at most 300 selected images and their annotations from one camera view across sunny, cloudy, and rainy conditions.

Compare two native Torchvision detectors without fine-tuning:

1. `ssdlite320_mobilenet_v3_large` as the lightweight baseline
2. `fasterrcnn_mobilenet_v3_large_fpn` as the accuracy-oriented baseline

Run both on CPU first. MPS is an optional measured experiment, not an assumed requirement. ONNX export and deployment are out of scope.

The detector-only approach is provisional. If distant or occluded cars are missed often enough to make per-space occupancy unreliable, pivot to a parking-space crop classifier based on MobileNetV3. That is a normal feasibility outcome, not a failed project.

## 2. Local constraints

- Apple M1 MacBook Air
- 8 GB unified memory
- Approximately 23 GiB free disk at research time
- Python 3.11 project environment
- No NVIDIA GPU
- Native PyTorch execution only
- Public portfolio use must remain possible
- Dataset and dependency licenses must be documented

These constraints rule out downloading multiple complete multi-gigabyte datasets or beginning with a heavyweight detector.

## 3. Dataset requirements

The first dataset must support most of the following:

- Fixed-camera parking viewpoint
- Known parking-space geometry
- Occupied or vacant labels
- Multiple distances from the camera
- Weather or illumination variation
- A temporal ordering or timestamps
- A license compatible with a public portfolio
- Selective download or a small local subset
- No requirement to redistribute sensitive source footage

No single reviewed dataset perfectly covers occupancy, continuous video, camera failures, weather, and unrestricted redistribution. The practical solution is to use licensed fixed-camera parking images and synthesize feed failures such as freezes, dropped frames, blur, and camera shifts.

## 4. Dataset comparison

### 4.1 PKLot

Verified properties:

- 12,417 parking-lot images at 1280 x 720
- Three camera views across two parking lots
- Sunny, cloudy, and rainy conditions
- XML parking-space coordinates and occupied or vacant labels
- Approximately 695,900 segmented parking-space examples
- Images captured at roughly five-minute intervals
- Full archive size of 4.6 GB
- CC BY 4.0 license with attribution and citation requirements

Strengths:

- Direct match for fixed-camera parking occupancy
- Existing space polygons eliminate unnecessary manual setup
- Weather labels support real distribution slices
- Chronological images can approximate occupancy and dwell transitions
- Clear source license

Limitations:

- Time-lapse images, not continuous video
- No genuine frame-rate, network, or frozen-stream failures
- Camera views are limited to three
- Original annotations describe spaces and occupancy, not standard vehicle bounding boxes
- Full download is too large for an initial spike on the current machine

Decision: **Primary feasibility dataset.** Use a selective subset, not the full archive.

Primary source:

- https://web.inf.ufpr.br/luizoliveira/research-interests/pklot/

### 4.2 MetaPKLot

Verified properties:

- Harmonizes PKLot, CNRPark-EXT, and PLds
- More than 2.2 million labeled samples after additions and corrections
- Standardized parking-space polygons
- Vehicle instance polygons
- Persistent vehicle identifiers supporting dwell-time experiments
- COCO-like JSON annotation structure
- Cross-dataset evaluation protocols
- Reference challenge code under the MIT license
- Source images retain their original dataset licenses

Strengths:

- Best current research reference for occupancy, dwell time, and parking-space extraction
- Provides stronger annotations than original PKLot
- Makes detector versus crop-classifier comparison easier
- Explicitly emphasizes cross-site evaluation

Limitations and license boundary:

- The dataset repository does not declare a single repository-level license in GitHub metadata.
- Its README identifies PKLot as CC BY 4.0 and CNRPark-EXT as ODbL 1.0.
- PLds requires access from its original authors.
- The MIT license on the challenge code does not automatically make every included image or annotation MIT licensed.
- The dataset repository is reported by GitHub at roughly 4.8 GB before Git history and checkout overhead.
- The challenge repository also contains large stored model and result artifacts and should not be cloned blindly.

Decision: **Use as a protocol and annotation reference.** For Phase 1, retrieve only PKLot-origin files whose source license is clear. Do not mix CNRPark-EXT or PLds into the project until their obligations are documented separately.

Primary sources:

- https://link.springer.com/article/10.1007/s00521-026-12398-0
- https://github.com/DSBD-Research/MetaPKLot-Dataset
- https://github.com/DSBD-Research/MetaPKLot-Challenges

### 4.3 CNRPark+EXT

Verified properties:

- Roughly 150,000 labeled parking-space patches
- 164 parking spaces
- Nine cameras in the EXT subset
- Multiple days and weather conditions
- Fixed smart-camera deployment context
- CNRPark-EXT is identified by the MetaPKLot authors as ODbL 1.0

Strengths:

- More camera diversity than PKLot
- Good later test of cross-camera generalization
- Closely related to edge parking occupancy research

Limitations:

- Primarily screenshots and cropped space patches rather than continuous video
- ODbL obligations require a separate compliance decision
- Not necessary for the first feasibility question

Decision: **Deferred cross-site evaluation dataset.** Do not download during the first spike.

Primary source:

- https://cnrpark.it/

### 4.4 VIRAT Ground Camera

Verified properties:

- Approximately 8.5 hours of stationary ground video
- Eleven outdoor scenes
- Multiple resolutions and frame rates
- Vehicle and activity annotations
- Natural surveillance conditions
- Lawful research and commercial use allowed under its signed agreement

Strengths:

- Genuine stationary surveillance video
- Useful for frame ingestion, motion, and temporal testing
- Includes parking-lot-like scenes and vehicle activity

Limitations:

- Every user must accept the usage agreement.
- Source data cannot be redistributed to third parties who have not accepted it.
- Users must protect against disclosure of personally identifiable information.
- It does not provide the parking-space occupancy geometry needed by this project.
- These restrictions complicate a public repository and shareable demonstration.

Decision: **Reject for the public MVP.** Reconsider only for private temporal research with proper agreement handling.

Primary sources:

- https://viratdata.org/
- https://viratdata.org/resources/VIRAT-Video-Data-Set-Protection-Agreement-1-4-11.pdf

## 5. Why PKLot images are sufficient for the first reliability experiment

PKLot does not provide real continuous video, so it cannot prove production stream reliability. It can still answer the first architectural questions:

1. Can a COCO-pretrained detector identify cars from high, oblique parking views?
2. Can detections be converted into correct per-space occupancy?
3. How does accuracy change across distance and real weather categories?
4. Do synthetic blur, darkness, obstruction, and camera shifts produce measurable degradation?
5. Can an abstention policy reduce incorrect confident occupancy decisions?

Chronological PKLot frames can be assembled into a controlled image sequence. Synthetic repeated frames and omissions can then test frozen-feed and dropped-frame logic without claiming that the sequence is real-time video.

## 6. Detector comparison

### 6.1 Torchvision SSDLite320 MobileNetV3 Large

Official Torchvision reference metrics on COCO val2017:

- 3.4 million parameters
- 0.58 GFLOPS
- 21.3 box mAP
- Fixed 320 x 320 model input

Advantages:

- Smallest reviewed detector
- Appropriate first speed and memory baseline for an 8 GB M1 machine
- Native PyTorch implementation
- Torchvision source code uses BSD 3-Clause

Risks:

- Lower COCO accuracy
- 320 x 320 input may lose distant vehicles in wide parking-lot frames
- Torchvision labels the detection module as beta

Decision: **Test as the lightweight baseline, not the assumed winner.**

### 6.2 Torchvision Faster R-CNN MobileNetV3 Large FPN

Official Torchvision reference metrics on COCO val2017:

- 19.4 million parameters
- 4.49 GFLOPS for the full FPN variant
- 32.8 box mAP

Advantages:

- Better reference accuracy than SSDLite
- Feature pyramid is more promising for vehicles at different apparent sizes
- Still substantially lighter than ResNet50-based Faster R-CNN
- Native PyTorch implementation

Risks:

- Higher memory and latency than SSDLite
- MPS operator support must be measured rather than assumed
- May still miss small or heavily occluded parked vehicles

Decision: **Primary accuracy-oriented detector candidate.**

### 6.3 Faster R-CNN ResNet50 FPN V2

Official Torchvision reference metrics:

- 43.7 million parameters
- 280.37 GFLOPS
- 46.7 box mAP

Decision: **Reject for the initial local spike.** The additional compute is disproportionate to the first feasibility question on an 8 GB laptop.

### 6.4 YOLOX

Verified properties:

- Apache 2.0 repository license
- Native PyTorch implementation
- Nano and Tiny variants available
- Last tagged GitHub release shown as April 2022

Advantages:

- Permissive source license
- Strong object-detection baseline family
- Lightweight variants

Risks:

- Older release and likely dependency friction with a current Python and PyTorch environment
- Adds a separate framework and preprocessing path before Torchvision feasibility is known

Decision: **Fallback candidate only.** Test it if both Torchvision candidates fail for model-quality rather than occupancy-mapping reasons.

Primary source:

- https://github.com/Megvii-BaseDetection/YOLOX

### 6.5 Ultralytics YOLO

Verified license boundary:

- Ultralytics distributes its current code and models under AGPL-3.0 or a commercial Enterprise license.
- Its own documentation says projects using the AGPL code or models must comply with the open-source requirements, while commercial closed use requires an Enterprise license.

Decision: **Do not use for this project now.** A public AGPL portfolio could be possible, but it would create unnecessary ambiguity if the work is later discussed with or adapted by a product company.

Primary source:

- https://docs.ultralytics.com/help/contributing/#open-sourcing-your-yolo-project-under-agpl-30

## 7. Model-license caution

Torchvision source code is BSD 3-Clause, but Torchvision explicitly warns that pretrained weights may inherit terms from their training datasets. The first experiment may use the official COCO weights for research and portfolio evaluation, but the project must record:

- Exact weight enum and checksum when available
- Torchvision and PyTorch versions
- COCO training provenance
- Source code license
- The distinction between portfolio research and any later company deployment

If SpotGenius or another company wants to reuse the work, model and weight licensing must be reviewed again for that specific use.

Primary sources:

- https://github.com/pytorch/vision/blob/main/LICENSE
- https://github.com/pytorch/vision
- https://docs.pytorch.org/vision/master/models.html

## 8. Detector versus parking-space classifier

The project should not assume that object detection is the optimal occupancy architecture.

### Detector approach

```text
full frame -> vehicle boxes -> box-to-space assignment -> occupancy
```

Strengths:

- Works without parking-specific training
- Produces interpretable vehicle locations
- Supports cars outside marked spaces

Weaknesses:

- Distant cars may be too small
- One box can overlap multiple spaces
- Generic COCO training does not optimize per-space occupancy

### Space-classifier approach

```text
full frame -> configured space crops -> occupied or empty classifier
```

Strengths:

- Directly optimizes the required output
- Can use PKLot and MetaPKLot labels naturally
- Recent MetaPKLot baselines use MobileNetV3 teacher models and lightweight student classifiers

Weaknesses:

- Requires parking-specific training or adaptation
- Does not directly localize arbitrary vehicles
- A camera shift corrupts the configured crops, making drift detection essential

Feasibility decision:

- Start with pretrained detectors because they require no training.
- If detector occupancy Macro F1 is below 0.85 or occupied-space recall is below 0.90 on the bounded evaluation set, test a MobileNetV3 space classifier before adding a third detector.
- Keep the camera-health and abstention layers independent of the selected occupancy method.

The thresholds above are project go or pivot criteria, not claims about industry requirements.

## 9. Selected feasibility experiment

### Dataset sample

- Dataset origin: PKLot only
- Initial camera: UFPR04
- Maximum initial sample: 300 images
- Weather balance: up to 100 sunny, 100 cloudy, and 100 rainy images
- Temporal selection: preserve filenames and timestamps
- Space selection: 20 spaces spanning near, middle, and far image regions
- Data transfer target: below 500 MB

Use the MetaPKLot GitHub repository only for selective file retrieval and annotation discovery. Do not clone its full approximately 4.8 GB dataset repository.

### Models

- SSDLite320 MobileNetV3 Large with official COCO weights
- Faster R-CNN MobileNetV3 Large FPN with official COCO weights

### Runtime matrix

- CPU is mandatory.
- MPS is tested only if the model executes without correctness or unsupported-operation problems.
- Batch size is one.
- Record warm-up separately from timed inference.
- Report median and p95 latency.

### Occupancy mapping candidates

Test two simple, explainable rules:

1. Detection-box center lies inside the parking polygon.
2. Intersection over parking-space area exceeds a configured threshold.

Do not add tracking, learned assignment, or complex heuristics until these baselines are measured.

### Evaluation slices

- Overall
- Occupied versus vacant
- Sunny, cloudy, and rainy
- Near, middle, and far spaces
- Clean versus synthetically degraded

### Metrics

- Per-space Macro F1
- Occupied-space precision and recall
- False state changes across chronological frames
- Percentage of outputs changed to `unknown`
- Incorrect confident outputs before and after abstention
- Median and p95 inference latency
- Peak process memory when practical

### Synthetic degradations

- Gaussian blur at several severities
- Brightness reduction
- Overexposure or glare approximation
- Partial obstruction masks
- Horizontal and vertical camera translation
- Small camera rotation
- Repeated frames
- Dropped frames

## 10. Phase 1 completion gates

Phase 1 is complete only when:

- Dataset source, license, citation, selected paths, and download date are recorded.
- No full multi-gigabyte dataset was downloaded unnecessarily.
- Both Torchvision detectors run on the same sample and hardware.
- Detector outputs are converted into per-space predictions.
- Metrics are computed against source labels.
- Results are split by weather and space distance.
- Failure injection demonstrates at least blur, darkness, and alignment shift.
- The project makes an evidence-based choice among detector-only, classifier pivot, or revised data.
- No ONNX work is added.

## 11. Immediate next implementation step

Write a dataset manifest and selective downloader for the chosen PKLot UFPR04 subset. The downloader must:

- Fetch only explicitly listed files
- Preserve original relative paths
- Record source URLs and timestamps
- Compute SHA-256 hashes
- Refuse to place data under Git tracking
- Produce a machine-readable manifest

After the data sample is verified visually, install PyTorch, Torchvision, OpenCV, and minimal measurement dependencies and run the two-detector smoke test.
