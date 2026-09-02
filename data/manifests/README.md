# PKLot UFPR04 manifests

This directory tracks metadata only. Original PKLot images remain local under ignored `data/raw/`.
The materialized 90-image working subset remains local under ignored `data/processed/`.

## Source

- Dataset: PKLot
- Camera view: UFPR04
- Publisher: Vision, Robotics and Image Laboratory, Federal University of Parana
- Archive: <https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz>
- Archive SHA-256: `e89bbc1dc735298c478688d50c7a682fb3b0076a87b6634923132709f2d2fa9b`
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)

Required citation:

> P. R. L. de Almeida, L. S. Oliveira, A. S. Britto Jr., E. J. Silva Jr., and
> A. L. Koerich. "PKLot - A robust dataset for parking lot classification."
> Expert Systems with Applications 42(11), 4937-4949, 2015.
> <https://doi.org/10.1016/j.eswa.2015.02.009>

## Reproduction

From the repository root, after extracting UFPR04 to `data/raw/PKLot/UFPR04`:

```bash
python3 -m pip install -e '.[dev]'
python3 -m scripts.inventory_pklot --strict
python3 -m scripts.select_ufpr04_subset
python3 -m scripts.split_ufpr04_phase1
python3 -m scripts.create_subset_contact_sheets
python3 -m scripts.verify_ufpr04_subset
python3 -m scripts.verify_ufpr04_phase1_splits
```

The selected source pool contains 90 full frames: 30 for each of Sunny, Cloudy, and Rainy.
Within each weather condition, it selects 10 frames from each of three acquisition dates. Selected
frames are at least 15 minutes apart. Dates and frames are selected deterministically from
chronological buckets using seed `20260902`.

Date selection chooses one date from each chronological bucket and minimizes the difference
between the selected occupancy rate and the full inventory's occupancy rate for that weather.
This preserves the source label distribution without forcing artificial 50/50 class balance.

The original plan called for a 30-minute gap, but only one rainy acquisition date contains ten
frames at that spacing. The 15-minute rule retains three rainy dates while keeping selected frames
three times farther apart than PKLot's five-minute capture interval.

This subset is balanced by weather, not by occupied/vacant parking-space labels. Images from the
same acquisition date must remain in one split if later work introduces training and test sets.
Some original UFPR04 XML files contain parking-space geometry without an `occupied` attribute.
The inventory records those as unlabeled spaces rather than treating the full frame as invalid.

## Phase 1 splits

`pklot_ufpr04_phase1_splits_v1.csv` assigns the source pool into three 30-image manifests:

- Smoke: 10 images per weather condition
- Calibration: 10 images per weather condition
- Held-out evaluation: 10 images per weather condition

Every split owns whole acquisition dates. No weather/date group appears in multiple splits. The
date assignment minimizes differences in occupancy rate across splits, using the recorded seed as
a deterministic tie-breaker. The current source pool has only three dates per weather, so a
12/48/30 allocation would require splitting dates and violate the leakage boundary.
