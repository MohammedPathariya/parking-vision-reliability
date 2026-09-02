# Project Scope

## Problem

Camera-based parking systems can produce incorrect occupancy results when the feed is frozen, degraded, obstructed, or misaligned. The project will determine both the parking-space state and whether the visual evidence is reliable enough to use.

## MVP contract

Given a fixed-camera recording and configured parking-space polygons, the system will eventually return:

- Per-space `available`, `occupied`, or `unknown` state
- Occupancy confidence
- Camera-health state
- Explainable reason codes
- Reliable dwell duration
- An `automation_allowed` decision

## Reliability principle

When camera evidence is unreliable, affected parking spaces must become `unknown`. The system must not convert every frame into a confident operational decision.

## Phase gates

### Phase 1: Dataset and detector feasibility

- Verify dataset licenses and attribution requirements.
- Select one bounded fixed-camera sample and reserve cross-camera data for later evaluation.
- Configure 10 to 30 parking spaces.
- Test candidate pretrained vehicle detectors.
- Record baseline accuracy, latency, and failure cases.

### Phase 2: Baseline occupancy

- Implement deterministic video decoding and sampling.
- Assign vehicle detections to parking polygons.
- Add temporal occupancy state and dwell tracking.
- Evaluate against a manually labeled subset.

### Phase 3: Camera reliability

- Detect missing, delayed, repeated, and frozen frames.
- Measure blur, darkness, contrast, and severe overexposure.
- Detect camera-position drift and major obstruction.
- Link failures to affected parking spaces.

### Phase 4: Failure injection and evaluation

- Introduce controlled image and stream degradation.
- Measure failure detection and occupancy degradation.
- Define evidence-based abstention thresholds.

### Phase 5: API and demonstration

- Expose typed health, occupancy, and event outputs.
- Add a minimal operational interface.
- Package the reproducible local demonstration.

## Current decision boundary

Do not select the production detector, video framework, dataset, dashboard technology, or deployment target until Phase 1 documents licensing, compatibility, and measured feasibility.
