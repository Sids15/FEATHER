# Conference introduction paper — FEATHER

A self-contained **introductory paper** (target ≤ 11 pages) for
conference/Review-I use: problem and stakes, drift background and taxonomy,
a survey of the three monitoring families, problem statement with
requirements, the FEATHER proposal (guiding hypothesis, a guarded
"glimpse" with diagrams, the synthetic-testbed results, research
questions), evaluation protocol, feasibility, risk assessment, expected
outcomes, and future work. It deliberately does **not** disclose the
method's construction or theory, and no deep-network results — those live
in the full manuscript in `../paper/`, which this folder never touches.

## Contents

```
paper-conference/
├── main.tex               ← the paper (Springer SNmult format)
├── references.bib         ← shared verified reference list (copy of ../paper/)
├── figures/
│   ├── drift_types.png    ← schematic; src/experiments/drift_types_figure.py
│   ├── geometry_toy.png   ← guarded; src/experiments/conference_figures.py
│   └── synthetic_demo.png ← guarded; src/experiments/conference_figures.py
├── SNmult.cls             ← Springer Nature class file
└── spmpsci.bst            ← numbered bibliography style
```

## Compiling

Upload the whole folder to [Overleaf](https://overleaf.com) and compile with
pdfLaTeX + BibTeX. `[TODO: …]` markers: author names/affiliations/emails and
the acknowledgement.

## Scope rule

Nothing in this paper should reveal more than: "we use the Fisher
information geometry of the deployed model to separate drift the model can
tolerate from drift it cannot see." No propositions, no algorithm, no
statistics definitions, no rank bound, no deep-network results.
**Allowed by decision (2026-07-17):** the synthetic 2-D testbed results
and the two guarded figures from `conference_figures.py` (sanitized
labels — no null(W), no statistic names, no subspace construction).
