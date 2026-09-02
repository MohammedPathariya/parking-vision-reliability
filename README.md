# Parking Vision Reliability

A work-in-progress computer-vision system for fixed parking-lot cameras that reports parking occupancy only when the camera evidence is reliable.

## The problem

Camera-based parking systems can be misled by blur, darkness, glare, obstructions, a shifted camera, or a frozen feed. Reporting a space as available when the camera is no longer trustworthy is worse than reporting that the system cannot determine its state.

## What this project will build

Given a fixed camera feed and configured parking-space regions, the system will:

- detect vehicles and estimate whether each space is occupied or available;
- monitor the camera feed for reliability issues such as blur, poor lighting, glare, obstruction, feed freezes, and viewpoint shifts;
- attach an explainable confidence assessment to each decision; and
- mark a space as `unknown` when the evidence is not reliable enough for automation.

For example:

```
Space A3: unknown
Reason: camera alignment shifted
Automation allowed: no
```

## Final goal

Build and evaluate a reliability-aware parking-vision pipeline that produces occupancy status, occupancy confidence, camera-health status, reason codes, and an explicit decision about whether automated use is allowed. The key outcome is not just detecting cars. It is knowing when the system should abstain rather than make a confident but incorrect parking decision.

## Scope

The project uses fixed-camera parking imagery and public or synthetic data to evaluate occupancy decisions under degraded camera conditions. It does not include license-plate recognition, facial recognition, payment systems, enforcement workflows, or customer data.

## Status

Phase 1 is dataset and pretrained-detector feasibility research. No model-performance claims have been made yet.

## License

This project is licensed under the [MIT License](LICENSE).
