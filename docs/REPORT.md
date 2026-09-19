# Multi-property QSPR predictor — results

Predicting aqueous solubility (`logS`), lipophilicity (`logP`, measured as logD at
pH 7.4) and melting point (`mp_K`) from SMILES, each with a calibrated 90%
prediction interval and an applicability-domain flag.

Everything below comes from one run on 15 September 2026, commit `6190d22`,
16 h 45 min on an M4 Mac mini. Reproduce with `experiments/run_pipeline.sh full`.

---

## 1. Headline

| Property | RMSE (95% CI) | RMSE ÷ SD | MAE | Spearman ρ | vs. predict-the-mean | n |
|---|---|---|---|---|---|---|
| `logS` log mol/L | 0.972 [0.865, 1.079] | **0.420** | 0.670 | 0.895 | 58% better | 9,383 |
| `logP` log units | 0.689 [0.624, 0.754] | **0.573** | 0.522 | 0.810 | 44% better | 4,200 |
| `mp_K` K | 41.34 [39.34, 43.35] | **0.452** | 31.04 | 0.881 | 55% better | 20,076 |

Intervals are t-intervals across the five outer folds. RMSE ÷ SD is the primary
number: 1.0 is the skill of predicting the training mean, and these sit at roughly
half that. Melting point is hardest in relative terms, as expected of a property
governed by crystal packing that SMILES does not encode.

Per-fold RMSE ÷ SD shows how much of that varies by which chemistry is held out:

| Property | fold 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| `logS` | 0.335 | 0.445 | 0.465 | 0.425 | 0.430 |
| `logP` | 0.652 | 0.536 | 0.581 | 0.529 | 0.568 |
| `mp_K` | 0.410 | 0.470 | 0.441 | 0.452 | 0.487 |

Fold 0 is the "benzene fold": one Bemis–Murcko group — a bare benzene ring — holds
6,137 molecules, 21% of the dataset. Scaffold groups are atomic, so that fold is
unavoidably larger, richer in melting points and poorer in lipophilicity. The
variation is reported rather than smoothed away.

---

## 2. Data

| Source | Property | Rows in |
|---|---|---|
| AqSolDB (authors' GitHub) | `logS` | 9,982 |
| Lipophilicity, AstraZeneca (DeepChem S3) | `logP` | 4,200 |
| Bradley open melting point (figshare 1031637) | `mp_K` | 28,645 |

Every raw file is MD5-verified on load, not only on download.

**After curation: 29,404 molecules**, with 816 rejections, each logged to a ledger
rather than dropped silently:

| Rejection | Count |
|---|---|
| Bradley "do not use" flag | 377 |
| Mixture (not a salt) | 251 |
| Duplicate key, collapsed to median | 152 |
| Outside physical range | 21 |
| Unparseable | 15 |

**Labels are sparse, and that sparsity is the premise of the project.** Only 174
molecules carry all three properties. Lipophilicity is nearly disjoint from the
rest: 207 molecules shared with solubility, 348 with melting point. Any multi-task
transfer for lipophilicity therefore has to come through the shared
representation, not through co-labelled rows.

Two curation rules are deliberate and easy to mistake for bugs. Fragments are
deduplicated *before* the mixture test, so a metal salt of two identical ligands is
stripped rather than rejected. And a SMILES that fails to kekulize gets one retry
with explicit hydrogens on aromatic nitrogens, which recovers about 280 otherwise
lost Bradley compounds.

---

## 3. Protocol

- **Scaffold splitting.** Bemis–Murcko groups are dealt whole into 5 outer × 3
  inner folds, balanced on per-property *label* counts rather than molecule counts.
  No test molecule shares a ring system with a training molecule. The 15% of
  molecules that are acyclic get singleton groups, which measured ~20× better
  solubility label balance across folds than pooling them.
- **Nested cross-validation.** 50 Optuna trials searched on inner folds only; the
  outer fold is scored exactly once, after every choice is made. This estimates the
  *procedure*, search included. Tree counts from inner early stopping are carried
  into the outer refit so that refit never sees the test fold.
- **Two tracks on identical folds.** Track 1 is per-property XGBoost over 217 RDKit
  descriptors plus 1,024 ECFP4 bits. Track 2 is a small multi-task head over a
  frozen transformer embedding, trained with a masked loss so each molecule
  contributes only the labels it has.
- **Metrics are never pooled across properties**, and coverage is never reported
  without interval width.

---

## 4. Model selection

Both tracks were scored on the same held-out molecules, so the comparison is
paired: per outer fold, the RMSE difference on identical rows, then a 95% interval
over the five differences. A property counts as a win only if that interval
excludes zero.

| Property | RMSE difference (XGBoost − multi-task) | Winner |
|---|---|---|
| `logS` | −0.218 [−0.270, −0.167] | XGBoost |
| `logP` | −0.146 [−0.172, −0.120] | XGBoost |
| `mp_K` | −6.88 K [−7.62, −6.14] | XGBoost |

**XGBoost won all three, decisively.** RMSE ÷ SD across tracks:

| Track | `logS` | `logP` | `mp_K` |
|---|---|---|---|
| XGBoost + descriptors | **0.420** | **0.573** | **0.452** |
| Multi-task, ChemBERTa-2 | 0.514 | 0.695 | 0.527 |
| Multi-task, PubChem10M | 0.532 | 0.748 | 0.525 |

**The encoder result is counter-intuitive and worth stating plainly.** ChemBERTa-2
ships a BPE tokenizer with zero merge rules: it reads `Cl` as `C` and `Br` as `B`,
and drops stereochemistry. It was pretrained that way, so the tokenizer cannot be
repaired after the fact — the multi-character embeddings sit near initialization
(norm ~2.35 against ~3.6 for trained tokens). A second encoder with a working
tokenizer was cached specifically so a neural-track loss could not be confounded
with halogen blindness. 31% of these molecules carry a halogen, and yet the
broken-tokenizer encoder scored *better* on every property. Halogen blindness was
not the limiting factor; the frozen embeddings are simply weaker features here than
descriptors plus fingerprints.

### Ranking quality

Screening cares about ordering, not absolute error. Enrichment factor at the top
10% (1.0 = random, 10.0 = perfect) and Spearman ρ within that decile:

| Property | EF@10% | top-decile ρ |
|---|---|---|
| `logS` | 7.06 | 0.531 |
| `logP` | 4.79 | 0.241 |
| `mp_K` | 5.95 | 0.374 |

Top-decile ρ is much lower than overall ρ for all three. That is largely
restriction of range: within a narrow slice, the remaining spread is closer to
noise. The enrichment factor, which ranks against the full list, is the more
honest screening metric.

---

## 5. Uncertainty

Raw uncertainty (XGBoost quantile regression) sets each interval's *shape*;
conformal calibration on out-of-fold residuals sets its *width*.

| Property | Method | Coverage (nominal 0.90) | Mean width |
|---|---|---|---|
| `logS` | CQR | 0.913 | 3.66 log units |
| `logS` | split conformal | 0.907 | 3.40 |
| `logP` | CQR | 0.918 | 2.70 log units |
| `logP` | split conformal | 0.909 | 2.48 |
| `mp_K` | CQR | 0.910 | 149 K |
| `mp_K` | split conformal | 0.909 | 143 K |

Marginal coverage is close to nominal, but that is nearly guaranteed by
construction and proves little. **Conditional coverage is the real test**, and it
degrades exactly where novelty is highest:

| Distance to nearest training molecule | `logS` | `logP` | `mp_K` |
|---|---|---|---|
| 0.00–0.20 (close analogues) | 0.943 | 0.947 | 0.953 |
| 0.20–0.35 | 0.929 | 0.944 | 0.912 |
| 0.35–0.50 | 0.923 | 0.925 | 0.916 |
| 0.50–0.65 | 0.899 | 0.895 | 0.899 |
| 0.65–1.00 (most novel) | **0.877** | **0.870** | **0.861** |

Intervals over-cover on familiar chemistry and under-cover on unfamiliar
chemistry. Mean width barely changes across those bands, so the intervals are not
adapting enough to novelty. A pooled 0.91 would have hidden this entirely.

### Applicability domain

Error was binned into ten equal-count bins by nearest-neighbour Tanimoto distance.
The threshold is the last distance before normalised error exceeds 1.5× the error
among the nearer half — read off the curve, never guessed. Per property: 0.654
(`logS`), 0.617 (`logP`), 0.900 (`mp_K`). A scored record carries one flag for the
whole molecule, so the strictest wins: **0.617**, which flags 13.5% of held-out
molecules as out of domain (20.0% of lipophilicity, 16.5% solubility, 10.7%
melting point).

---

## 6. Low-data ablation

The hypothesis under test: multi-task learning should help most when labels are
scarce, since a property can borrow strength from the others through the shared
representation. Labels were subsampled to N per property inside each training fold,
with fixed hyperparameters for both tracks, scored on the full held-out fold.

RMSE ÷ SD, mean over 3 seeds × 5 folds:

| Property | N | XGBoost | Multi-task | Multi-task advantage |
|---|---|---|---|---|
| `logS` | 100 | 0.660 | 0.803 | −0.143 |
| `logS` | 250 | 0.590 | 0.702 | −0.112 |
| `logS` | 500 | 0.553 | 0.648 | −0.095 |
| `logS` | 1000 | 0.508 | 0.631 | −0.123 |
| `logP` | 100 | 0.917 | 0.978 | −0.061 |
| `logP` | 250 | 0.829 | 0.923 | −0.094 |
| `logP` | 500 | 0.742 | 0.877 | −0.135 |
| `logP` | 1000 | 0.671 | 0.825 | −0.155 |
| `mp_K` | 100 | 0.721 | 0.767 | −0.046 |
| `mp_K` | 250 | 0.648 | 0.701 | −0.052 |
| `mp_K` | 500 | 0.602 | 0.667 | −0.065 |
| `mp_K` | 1000 | 0.565 | 0.642 | −0.076 |

**The hypothesis is refuted.** Multi-task lost at every size and every property.
For lipophilicity the gap *widened* with more data, the opposite of the predicted
shape, which is consistent with lipophilicity being nearly disjoint from the other
two: it has little to borrow, and pays the cost of sharing capacity.

One caveat: the ablation used the PubChem10M encoder, which the full comparison
rated the weaker of the two. Rerunning it on ChemBERTa-2 would test the claim with
the better representation.

---

## 7. What ships

A single MLflow pyfunc, registered as version 1 of `dupont-qspr-oracle`,
`model_version` `xgb-full-6190d22-20260915`, fit on all 29,404 molecules after a
fresh 50-trial search scored by cross-validation over the outer folds. Conformal
offsets come from out-of-fold predictions over those folds, so every molecule
contributes a residual from a model that never saw it: 0.168 (`logS`), 0.366
(`logP`), 14.51 K (`mp_K`).

Scoring path: SMILES → standardisation → featurization → point prediction and raw
quantiles → conformal interval → applicability-domain distance → scored record. The
interface takes a list, and an unparseable input yields a structured error record
rather than an exception, so one bad SMILES cannot kill a batch.

```json
{
  "smiles_canonical": "CC(=O)Oc1ccccc1C(=O)O",
  "predictions": {
    "logS": {"value": -1.83, "lower": -3.05, "upper": -0.54, "nominal_coverage": 0.9},
    "logP": {"value": -0.29, "lower": -1.59, "upper":  1.74, "nominal_coverage": 0.9},
    "mp_K": {"value": 393.4, "lower": 342.7, "upper": 468.0, "nominal_coverage": 0.9}
  },
  "applicability_domain": {"nn_tanimoto_distance": 0.0, "in_domain": true},
  "model_version": "xgb-full-6190d22-20260915",
  "featurizer_version": "v1:rdkit-ecfp4"
}
```

**One family ships rather than a per-property hybrid.** XGBoost and the transformer
encoder link separate OpenMP runtimes and crash when loaded into one process on
this machine, even loading the encoder first, so an in-process hybrid is not
buildable here. XGBoost winning all three properties made the point moot.

---

## 8. Cost

| Stage | Wall clock |
|---|---|
| Data, features, folds | seconds warm, ~15 min cold |
| XGBoost nested search | 9.7 h |
| Multi-task nested search, both encoders | 3.2 h |
| Conformal intervals + AD distances (×3 tracks) | 1.3 h |
| Analysis (family choice, AD threshold) | seconds |
| Low-data ablation | 4 min |
| Final search, calibration, registration | 2.5 h |
| **Total** | **16 h 45 min** |

The XGBoost search was projected at ~3 h from single fit times and took 9.7 h. The
projection assumed later trials cost what early random ones did, but TPE converges
on low learning rates that build far more trees before early stopping halts them;
the slowest single cell ran 69 minutes. Future budgets should be sized from the
measured figure.

---

## 9. Limitations

- **The applicability domain is drug-like chemical space.** Transfer to
  electronic-materials chemistries is unvalidated and is not claimed.
- **Intervals under-cover on novel chemistry** (0.86–0.88 against a nominal 0.90 in
  the most distant band) and their width barely adapts. The domain flag marks those
  molecules, but the interval itself does not keep its promise there.
- **`logP` is logD at pH 7.4**, which differs from true logP for ionisable
  molecules.
- **Melting point is intrinsically hard**: it depends on crystal packing, which
  SMILES does not encode.
- **Measurement heterogeneity sets a floor.** AqSolDB aggregates nine sources and
  Bradley is literature-compiled. Median replicate disagreement is 0.12 log units
  for solubility and 2 K for melting point — far below the errors above, so the
  current limit is the model, not the data.
- **Selection used held-out scores.** Choosing the family from outer-fold results is
  mild selection on held-out data; two candidates per property rather than fifty
  configurations, so the optimism is small but not zero.
- **Calibration is CV-recycled split conformal**, not strict CV+/jackknife+, so it
  lacks CV+'s formal finite-sample guarantee.
- **No published-leaderboard comparison.** PyTDC cannot be installed alongside
  current numpy, pandas and RDKit, so splits were rebuilt from primary sources and
  do not line up with published benchmark splits.
- **Frozen encoders, by choice.** With a few thousand labels per property,
  fine-tuning would overfit, and it would also invalidate the embedding cache the
  protocol's cost depends on.
- **No generation, optimisation or synthesizability scoring** — out of scope.

---

## 10. Reproducing

```sh
uv sync
experiments/run_pipeline.sh full
```

Each stage is a separate process that reads what the previous one persisted, so
stages can be rerun individually. Every run records its git revision, package
versions and seeds. Raw outputs: `artifacts/full/decision.json` (family choice,
comparison table, AD thresholds), `artifacts/full/runs/` (per-stage metrics),
`artifacts/full/figures/` (skill, parity, ranking, coverage-by-distance, ablation).
