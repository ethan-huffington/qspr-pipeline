# Documentation

`REPORT.md` is the written results report for the full run: headline numbers,
model selection, uncertainty, the low-data ablation, and limitations.

Everything else here is regenerated from the source tree, not hand-maintained.
Rerun the commands below after adding or moving a module.

## Module dependency graph

| File | What it is |
|---|---|
| `module-graph.svg` | The one to look at. Implementation modules only. |
| `module-graph-full.svg` | Same, plus the package `__init__` re-export shims. |
| `module-graph.dot` | Graphviz source for `module-graph.svg`. |
| `module-deps.json` | Raw dependency data, for scripting. |
| `module-layers.txt` | Text view, layered by dependency depth. |

**Arrows mean "depends on".** An arrow from `config` to `contracts` says config
imports contracts. This requires `--reverse`: pydeps' default draws the opposite
direction, meaning "is imported by", which reads backwards for a graph labelled
as dependencies.

### Regenerating

Requires `graphviz` for rendering (`brew install graphviz`); pydeps itself is a
dev dependency.

```sh
uv run pydeps src/dupont_qspr \
    --only dupont_qspr --max-bacon 0 --reverse \
    --exclude-exact dupont_qspr dupont_qspr.analysis dupont_qspr.data dupont_qspr.features \
        dupont_qspr.metrics dupont_qspr.models dupont_qspr.reporting dupont_qspr.serving \
        dupont_qspr.splits dupont_qspr.tuning dupont_qspr.uncertainty \
    --rmprefix dupont_qspr. --noshow --no-config \
    -T svg -o docs/module-graph.svg
```

Flag by flag:

- `--only dupont_qspr` — local code only; drops numpy, torch, rdkit and the rest
- `--max-bacon 0` — no depth limit (the default of 2 truncates the graph)
- `--reverse` — arrows point from dependent to dependency, as explained above
- `--exclude-exact …` — drops the package `__init__` modules without dropping
  their contents. Those files only re-export, so they add edges that say nothing
  about the design and make everything appear to depend on everything
- `--rmprefix dupont_qspr.` — shortens node labels
- `--noshow` — don't open a viewer
- `--no-config` — ignore any user-level `.pydeps` file, so output is reproducible

Drop `--exclude-exact` for `module-graph-full.svg`. Append `--show-dot --no-output`
(redirected to `module-graph.dot`) or `--show-deps --no-output` (to
`module-deps.json`) for the other two files. zsh does not word-split an unquoted
variable, so write the module list out rather than storing it in one.

### What the graph should show

Five layers, no cycles. Full listing in `module-layers.txt`.

```
layer 0   contracts · data.sources · features.cache · metrics.point · splits.scaffold
          tuning.spaces · uncertainty.applicability · uncertainty.conformal
layer 1   config · data.download · data.standardize · metrics.baselines
          models.mtl · models.xgb · reporting.figures
layer 2   analysis.* · data.curate_mp · data.union · dataset · features.*
          models.ensemble · reporting.load · reporting.tables · splits.nested · tracking
layer 3   analysis.ablation · data.build · serving.scorer · tuning.nested_*
layer 4   serving.final_fit · serving.pyfunc
```

`contracts` at layer 0, depended on by 19 modules, is the design working as
intended: the interfaces depend on nothing and everything depends on the
interfaces. If `contracts` ever acquires an outgoing edge, something has leaked an
implementation detail into the seam.

### Reading it with the OpenMP constraint in mind

pydeps follows imports written inside functions too, so `serving.scorer` shows
edges to **both** `models.xgb` and `models.mtl`. That is correct as a dependency
graph, but it does not mean both load at runtime. Each import sits inside the code
path for its own family, which is how one module serves either family without
XGBoost and PyTorch ever sharing a process. The same applies to
`analysis.ablation`.
