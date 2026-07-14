# FEATHER — Project Schedule (Gantt Chart)

Semester plan, July–December 2026. The Mermaid block below renders on GitHub;
rebuild it in Excel/PowerPoint for the slide if the panel prefers.

```mermaid
gantt
    title FEATHER — Capstone Schedule (Jul–Dec 2026)
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Phase 0 — Done early
    Repo, rules, venv, plan        :done, p0a, 2026-07-14, 2026-07-16
    Tier-1 core (Fisher+monitor)   :done, p0b, 2026-07-14, 2026-07-18

    section Review-I
    Lit review, slides, forms      :active, r1a, 2026-07-14, 2026-07-19
    Guide cross-check + Review-I   :crit, r1b, 2026-07-20, 2026-07-25

    section Phase 1 — Tier 2 (Aug)
    PyTorch env + MNIST CNN        :p1a, 2026-08-01, 2026-08-10
    Rotated-MNIST drift streams    :p1b, 2026-08-08, 2026-08-18
    FEATHER on Rotated MNIST       :p1c, 2026-08-15, 2026-08-31

    section Phase 2 — Baselines & Cascade (Sep)
    Output baselines (conf/ATC/BBSD):p2a, 2026-09-01, 2026-09-12
    PCA/Mahalanobis ablation       :p2b, 2026-09-08, 2026-09-18
    River stage-1 + FEATHER-Lite   :p2c, 2026-09-15, 2026-09-30

    section Phase 3 — Tier 3 + Demo (Oct–Nov)
    ResNet-18 + CIFAR-10-C sweep   :p3a, 2026-10-01, 2026-10-25
    Silent-drift headline experiment:crit, p3b, 2026-10-15, 2026-11-05
    Multi-seed runs + statistics   :p3c, 2026-10-25, 2026-11-15
    Streamlit dashboard            :p3d, 2026-11-01, 2026-11-20

    section Phase 4 — Paper & Close (Nov–Dec)
    Paper draft                    :p4a, 2026-11-10, 2026-12-05
    Docs, repo polish, demo video  :p4b, 2026-11-25, 2026-12-10
    Final review preparation       :crit, p4c, 2026-12-05, 2026-12-15
```

## Table form (for the slide)

| Phase | Window | Deliverable | Status |
|---|---|---|---|
| 0. Foundation | Jul 14–18 | Repo, rules, venv, revised plan, **tested Tier-1 core (38 tests)** | ✅ done |
| Review-I | Jul 20–25 | Forms, literature review, slides, this chart | 🔵 in progress |
| 1. Tier 2 | Aug | CNN on MNIST; Rotated-MNIST drift streams; FEATHER validation | planned |
| 2. Baselines | Sep | Confidence/ATC/BBSD + PCA ablation; River stage-1; FEATHER-Lite cascade | planned |
| 3. Tier 3 + demo | Oct–Nov | ResNet-18 + CIFAR-10-C sweep; **silent-drift headline experiment**; multi-seed stats; dashboard | planned |
| 4. Paper | Nov–Dec | Paper draft, repo polish, demo video, final review | planned |

**Buffer:** each phase ends with slack before the next begins; the dashboard and
paper depend only on Phase 1–2 outputs, so a Phase-3 overrun degrades scope
(fewer corruptions/seeds), never the deliverable. Milestones to compare against
Reviews II/III as required by the guidelines.
