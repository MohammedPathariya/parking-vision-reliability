# Parking Vision Reliability

A production-oriented computer-vision project for measuring parking-space occupancy while determining whether the camera evidence is trustworthy enough to use.

## Status

Initial project scaffold. Dataset, detector, and video-processing dependencies will be selected during the feasibility phase after licensing and compatibility checks.

## Core output

For each configured parking space, the system will eventually report:

- `state`: `available`, `occupied`, or `unknown`
- Occupancy confidence
- Camera-health state and reason codes
- Dwell time when reliably occupied
- Whether downstream automation is allowed

## Initial scope

- One fixed camera or recorded video
- A bounded set of configured parking-space polygons
- Vehicle detection and temporal occupancy state
- Feed, image-quality, and camera-alignment diagnostics
- Reliability-aware abstention
- Reproducible evaluation under controlled degradation

## Non-goals for the MVP

- License-plate or facial recognition
- Payment, permit, or enforcement integrations
- Multi-site cloud infrastructure
- Mobile applications
- Claims about SpotGenius systems or customer environments

## Local setup

The project targets Python 3.11 through 3.13. Python 3.11 is the initial development version because computer-vision and inference packages may not yet support the locally installed Python 3.14.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Run the initial checks:

```bash
python -m pytest
python -m ruff check .
```

## Repository layout

```text
configs/                         Versioned camera and experiment configuration
data/                            Data boundaries and provenance documentation
docs/                            Scope, decisions, and evaluation documentation
src/parking_vision_reliability/  Application package
tests/                           Automated tests
```

See [docs/PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md) for the current project contract and phase gates.
