# Conference introduction paper — FEATHER

A short, self-contained **introductory paper** for conference/Review-I use:
problem, motivation, gap, project objectives, planned evaluation, and
future work. It deliberately does **not** disclose the method's
construction, theory, or experimental results — those live in the full
manuscript in `../paper/`, which this folder never touches.

## Contents

```
paper-conference/
├── main.tex        ← the short paper (Springer SNmult format)
├── references.bib  ← shared verified reference list (copy of ../paper/)
├── SNmult.cls      ← Springer Nature class file
└── spmpsci.bst     ← numbered bibliography style
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
