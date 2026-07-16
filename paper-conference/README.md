# Conference introduction paper — FEATHER

A self-contained **introductory paper** (~10–12 pages) for
conference/Review-I use: problem and stakes, drift background and taxonomy,
a survey of the three monitoring families, problem statement with
requirements, the FEATHER proposal (guiding hypothesis + research
questions only), work plan, evaluation protocol, feasibility, risk
assessment, expected outcomes, and future work. It deliberately does
**not** disclose the method's construction, theory, or experimental
results — those live in the full manuscript in `../paper/`, which this
folder never touches.

## Contents

```
paper-conference/
├── main.tex             ← the paper (Springer SNmult format)
├── references.bib       ← shared verified reference list (copy of ../paper/)
├── figures/
│   └── drift_types.png  ← schematic; src/experiments/drift_types_figure.py
├── SNmult.cls           ← Springer Nature class file
└── spmpsci.bst          ← numbered bibliography style
```

## Compiling

Upload the whole folder to [Overleaf](https://overleaf.com) and compile with
pdfLaTeX + BibTeX. `[TODO: …]` markers: author names/affiliations/emails and
the acknowledgement.

## Scope rule

Nothing in this paper should reveal more than: "we use the Fisher
information geometry of the deployed model to separate drift the model can
tolerate from drift it cannot see." No propositions, no algorithm, no
statistics definitions, no measured results, no figures from the runs.
