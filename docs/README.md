# Generated documentation

Everything here is regenerated from the source tree, not hand-maintained. Rerun
the commands below after adding or moving a module.

## Module dependency graph

| File | What it is |
|---|---|
| `module-graph.svg` | The one to look at. Implementation modules only. |
| `module-graph-full.svg` | Same, plus the four package `__init__` re-export shims. |
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
    --exclude-exact dupont_qspr dupont_qspr.data dupont_qspr.features dupont_qspr.splits \
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

Drop `--exclude-exact` for `module-graph-full.svg`.

### What the graph should show

Four layers, no cycles.

```
layer 0   contracts · data.sources · features.cache · splits.scaffold
layer 1   config · data.download · data.standardize
layer 2   data.curate_mp · data.union · features.descriptors · features.encoders
          skeleton · splits.nested · tracking
layer 3   data.build · spine
```

`contracts` sitting at layer 0 with nothing beneath it, and seven modules above
depending on it, is the design working as intended: the interfaces depend on
nothing and everything depends on the interfaces. If `contracts` ever acquires an
outgoing edge, something has leaked an implementation detail into the seam.

### Known limitation

pydeps analyses imports statically, so it misses imports written inside
functions. There is one in this codebase: `dupont_qspr/__init__.py` imports
`spine` inside `main()`, to keep `import dupont_qspr` cheap. That edge appears in
no pydeps output.
