# Verify — recompute the paper's numbers

Every number in the paper, recomputed from the shipped data and printed beside
what the paper says.

```
# Linux and Windows: install the CPU build of torch first,
# so pip does not pull several GB of CUDA libraries
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python verify.py            # everything except the second encoder
python verify.py --full     # also the second encoder
```

On macOS the plain `pip install -r requirements.txt` is already CPU-only.

**With an NVIDIA GPU**, install the CUDA build of torch for your driver instead of
the CPU line (see pytorch.org); verify then uses the GPU automatically and the
whole run takes a few minutes. It adds a few GB of CUDA libraries, which is why
CPU is the default. Numbers agree within every tolerance either way.

No GPU is needed. The default run takes roughly 15–30 minutes on a laptop, almost all of
it embedding about 150,000 generations once; `--full` re-embeds the steering
generations with a larger encoder and adds roughly 20–60 minutes. With
`--cache DIR` the embeddings are kept, and a rerun takes a few minutes. The only
downloads are the sentence encoders, fetched once by `sentence-transformers`.
Tested in a clean container: every check passes.

## What you get

```
CLAIM                                            PAPER   RECOMPUTED     TOL   STATUS
Gram mean |cos|                                  0.187        0.187   0.005   PASS
Gram condition number                            5.489        5.489     0.1   PASS
word-contrast null p95                           0.851        0.851    0.02   PASS
model-level mean gain                            0.405        0.405    0.02   PASS
diagonal is row max (ROWS = CONSTRUCT), mean     0.857        0.857   0.015   PASS
  symmetric off-diag change                     -0.236       -0.236    0.02   PASS
diag-is-max at one common magnitude, mean        0.540        0.540   0.002   PASS
```

The negative results are checked with the same tolerances as the positive ones:
the dual basis's failure to help when dials are combined, and the asymmetry no
geometric correction reaches. A package that only reproduced favourable numbers
would not be worth running.

## Sections

`python verify.py --list` prints them; `--only` runs a subset.

| section | what | needs |
|---|---|---|
| `geometry` | Gram matrix, leak, iso-cost, request cost, Gram–Schmidt, the alternative seven | vectors |
| `nulls`, `validation` | the two nulls; leave-one-word-out; pre-registered unseen words | vectors |
| `edges` | each c\* re-derived from its recorded ladder | `edges.json` |
| `sharedwords` | words shared between lists and their share of the Gram, with its null | vectors |
| `interface` | the two requests of section 4: coordinates, cost, top-ten tokens, nearest named emotions | atlas |
| `linear` | what a perfectly linear model shows in this pipeline | vectors, encoder |
| `geompred` | the single-dial gain geometry alone predicts, per model | vectors, encoder |
| `steering`, `robustness`, `crossboot` | gains over matched controls; raw, deduplicated and coherent text; the cross-alignment bootstrap | generations |
| `control`, `magnitude`, `spread` | transfer matrices per repetition and pooled; how far text moves | generations |
| `superposition` | single dials predicting three; the dual basis one at a time and three at once | both generation sets |
| `commonmag` | the same simultaneous-control experiment at one common magnitude, c = 0.08: 54% against 86% | `common_magnitude_transfer.json` |
| `encoder2` (`--full`) | the steering measurement repeated with bge-large-en-v1.5 | generations |

## Two row conventions

A transfer matrix can be read with rows as the dial turned or as the construct
measured, and both are in use, as they are in the analysis scripts the paper
quotes. The per-repetition and pooled matrices (`control`, `magnitude`) are
fitted with rows as constructs, so "the diagonal is the row maximum" asks
whether each construct is moved most by its own dial. The superposition
matrices have rows as dials. Each check's label and docstring say which.

## What a failure means

Not necessarily an error in the paper. Tolerances are in `expected.json`, with
the reason for each; a value just outside one is usually a library difference —
BLAS ordering, a different scikit-learn — while a value far outside is not.
`verify.py` prints the recomputed figure so the two can be told apart. The
sentence encoders are the components not under a seed: a different version
shifts every alignment number slightly, so `requirements.txt` pins the versions
that produced the published values.

## What this does and does not verify

**Verified.** The geometry of the basis, the axis validation, the steering
measurements, simultaneous control, the dual-basis comparison, the
superposition test, the linear baseline, and the interface readouts —
everything computed from the shipped vectors, generations and atlas.

**Not verified.** The extraction that produced the vectors and the generation
that produced the texts; both need model weights and a GPU. The coherence edges
are checked against their recorded ladders rather than recomputed. The 54% at a
single common magnitude (sections 3.5 and 7.2) is checked from that run's stored
transfer matrices; its generations are in the reproduction archive, not here.
**This package verifies the analysis, not the experiment.** The reproduction
archive covers the rest.

## What is in `data/`

| | |
|---|---|
| `vectors_layer/` | one file per model, the selected layer only. 1.7 MB against 46 MB for all layers, and no claim depends on any other layer. |
| `heldout_vectors/` | the pre-registered unseen words, extracted separately so neither the fit nor the axes have met them |
| `generations/` | the three-dial experiment: 90,720 steered texts in raw, dual and matched-control arms, gzipped |
| `generations_single/` | the single-dial experiment of section 7.3: the same trials and stems, one dial turned per cell, raw and dual arms, gzipped. Same file names and trial ids as `generations/`, hence its own folder |
| `common_magnitude_transfer.json` | the archived run at one common magnitude (c = 0.08 for every model): its fitted per-repetition transfer matrices |
| `atlas_g31bit.json.gz` | the Gemma-3-1B-it atlas the interface displays: the Gram matrix, token coordinates (int8) and named-emotion positions |
| `edges.json` | the nine coherence ladders and the c\* each yields |
| `layer_selection.csv` | per-layer held-out AUC. Single-layer vectors make the layer choice unrecomputable, so the table that made the choice ships instead |
| `probes.json` | all eighteen word lists |
| `predictions.csv` | the pre-registered unseen words, `word,probe,side` |
| `labels.csv` | EMO/NONEMO per sense |

`MANIFEST.json` has the sha256 of every file.

## The axis renamed between data and paper

The data calls one axis `social_inner_outer`; the paper calls it
isolation–belonging. Same word lists, same direction.

## Related

- **Interface** — a control surface you can open in a browser: seven dials, a
  logit-lens readout, and a toggle between raw and dual steering.
- **Reproduce** — the full archive: corpus, all layers, every run, and the
  staged pipeline that rebuilds everything from prompts.

## Licence

Code MIT; data CC BY-SA 4.0, because Wiktionary glosses are part of every sense
key and share-alike follows them. WordNet and Wiktionary notices are in
`LICENCE`. Model weights are not redistributed. The NRC VAD Lexicon is not
redistributed and no comparison against it appears here.
