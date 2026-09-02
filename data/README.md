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
