# Data boundary

Large datasets and generated media are not committed to Git.

Every dataset used by the project must have a documented:

- Source URL
- License and permitted use
- Download or access date
- File inventory or manifest
- Integrity hash when practical
- Annotation procedure
- Privacy review

Place downloaded source data under `data/raw/` and generated derivatives under `data/processed/`. Both directories are ignored except for their placeholders.

Current local layout:

- `data/raw/PKLot/UFPR04/`: the verified original UFPR04 JPEG/XML source inventory.
- `data/processed/ufpr04_balanced_v1/`: the 90-image weather-balanced working subset.
- `data/manifests/`: tracked inventory, subset manifest, provenance, and visual-QA record.

The Phase 1 experiment uses only the 90-image subset. The larger UFPR04 source copy is retained
locally to reproduce and audit the tracked inventory; it must never be committed.
