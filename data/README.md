# Data

The datasets are not stored in this repository. Each folder here keeps only the
`meta.json` written by the generator, which records the exact settings used
(angular range, number of projections, noise level, seed).

## Synthetic datasets

All `pilot_*` and `phys_*` sets are produced by `generate_dataset.py`. The
command for each one is in the batch file that ran the experiment (see the
experiment table in the main README). For example, the baseline set is:

```
python generate_dataset.py --n 800 --out data/pilot_v1
```

Each dataset folder then contains `inputs/` (limited-angle FBP images),
`targets/` (ground-truth phantoms), `meta.json` and `preview.png`.

## Real data (LoDoPaB-CT)

`e6_lodopab_test` and `e6b_lodopab_train` are built from the ground-truth
slices of the LoDoPaB-CT benchmark (Leuschner et al., 2021, Zenodo record
3384092). Download `ground_truth_test.zip` and `ground_truth_validation.zip`
from Zenodo, extract them into `data/lodopab_raw/`, then run
`build_e6_dataset.py` (or `scripts\run_e6.bat` and `scripts\run_e6b.bat`).
LoDoPaB-CT is derived from the LIDC-IDRI collection. Please follow the licence
terms given on the Zenodo record when using it.
