# UFPR04 balanced v1 visual QA

- Review date: 2026-09-02
- Manifest SHA-256: `d00257b4b0b6f6685cf86caa45846e583fb53772e6fd8167cefb330902abcb14`
- Frames reviewed: 90 of 90
- Result: pass
- Frames rejected: 0

## Contact sheets reviewed

- Sunny: `artifacts/qa/ufpr04_balanced_v1_sunny.jpg`
  - SHA-256: `3e88027d5feadfa5fd5ac215b3b8aa01e17caba2bc9699b20efc90b56d32bb31`
- Cloudy: `artifacts/qa/ufpr04_balanced_v1_cloudy.jpg`
  - SHA-256: `e23fc12b8edb93b9d0a94c9344fbe490662ac5ddaa9c36a60c7981e8eae4b825`
- Rainy: `artifacts/qa/ufpr04_balanced_v1_rainy.jpg`
  - SHA-256: `a1831763be4e7d9e86c24cef3558095af4927b16da559b90ab1deda19ef81e88`

## Checks

- All selected JPEGs render correctly.
- Parking-space contours align with the visible parking geometry.
- Occupied, vacant, and unlabeled overlays agree with the source annotations.
- Each weather subset includes visible variation in illumination and parking occupancy.
- No selected frame is fully obscured or unusable.

## Limitations observed

- Weather labels are inherited from the source directories. Some Rainy frames show wet pavement or
  post-rain lighting rather than visible falling rain.
- Parking occupancy is correlated with acquisition date. The subset preserves the full inventory's
  per-weather occupancy rates, but it is not suitable for random frame-level train/test splitting.
- Yellow contours represent source spaces with geometry but no `occupied` attribute. The selected
  subset contains eight such spaces across 90 frames.
- PKLot is time-lapse imagery and cannot validate continuous-video behavior such as dropped frames,
  frozen feeds, or exact state-transition delay.
