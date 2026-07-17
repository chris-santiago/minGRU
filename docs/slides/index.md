# Slides

Two presentation decks accompany the project. They open full-screen in the browser with arrow-key navigation (built with Marp; press `f` for fullscreen, `o` for overview).

- [**Parallel GRUs**](parallel-grus-deck.html) — the main deck: why minGRU trades gate expressivity for parallel-scan training, the state-tracking ladder (signed, rotation, Givens mixers), and the measured evidence behind each rung.
- [**Parallel GRUs — appendix**](parallel-grus-appendix.html) — supporting material: derivations, task constructions, and the full result tables behind the main deck's claims.

The decks are presentation artifacts and predate the packaged library: code shown uses the repo-checkout style (`from min_gru import ...`), which still works from a clone via the root evidence drivers; installed users import `mingru` as shown in the [tutorials](../tutorials/getting-started.md). The quantitative claims trace to `experiments/TECHNICAL_REPORT.md` and `experiments/EXPERIMENTS.md`.
