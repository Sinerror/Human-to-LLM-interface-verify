# Interface — an emotion control surface in the browser

Seven dials, each defined by a pair of contrasting emotion word lists, over a
language model's activations. Turn a dial and the page shows where the request
lands on every axis, which tokens the displacement promotes, and which named
emotions the state sits near. It runs in a browser without the model. This is
the prototype of section 4 of the paper.

![The control surface for Gemma-3-1B-it, in dual mode, for a request pleasant and hostile at once](docs/interface.png)

## Open it

**Live:** <https://sinerror.github.io/Human-to-LLM-interface/> — all nine models,
nothing to install.

**Offline:** bake a page (below) and open it by double-clicking. No server: the
page carries its data inside it.

## What the page shows

**Dials.** Seven bipolar axes — valence, heat, arousal, intensity,
antagonism–peace, body–mind, isolation–belonging — each labelled at both ends
with the words that define it.

**Raw / dual.** The axes overlap, as emotion constructs do. Under raw steering
a request `l` arrives as `G l`, so every off-diagonal entry of the Gram matrix `G`
shows up as movement nobody asked for. Under the dual basis `D = G⁻¹P` every
coordinate arrives exactly as requested, at a cost in displacement the page
reports for the current request: between 0.68 and 1.60 times what an orthogonal
basis would need, depending on whether the request runs with or against the way
the axes overlap.

**Where the cursor lands.** What a probe along each axis would register for the
displacement: the requested value beside the delivered one.

**Tokens this displacement promotes.** A logit-lens readout: each token's
unembedding projected onto the displacement, ranked. Tokens and displacement are
both expressed through the seven oblique directions, so the ranking uses the
metric `G⁻¹`. Some promoted tokens lie outside the affective field; they may be
noise or associations the model attaches to that position, and a logit lens
cannot tell the two apart.

**Nearest named emotion.** Ekman's six categories placed in the same
coordinates, with the cosine to the current state under the same metric.
Categories are unipolar, so none is the opposite of another: the panel shows
proximity, not classification.

**Basis geometry.** The model's Gram matrix, mean off-diagonal and condition
number.

## What it does not show

Every readout is linear: the first-order effect of the displacement at the layer
where it is applied. In that setting the dual correction is exact by
construction, and the page shows it working perfectly. Generated text responds to
the displacement linearly too, but not through this geometry: the model's
response is asymmetric where the geometry is symmetric, and in generation the
correction helps one dial at a time and much less when several are combined at a
fixed strength (paper, section 7.3). Treat the page as a picture of the geometry,
not as a preview of generated text.

## Atlases and baking

An atlas holds one model's seven directions, their Gram matrix, a coordinate for
each word-like vocabulary token, and the Ekman category positions. Atlases for
all nine models of the paper are precomputed in `atlases/`.

**All nine models in one page**, with a model selector — the most informative
way to use it, since the same seven word lists give a differently shaped basis in
every model:

```
python bake_interface.py --template index.html --atlas "atlases/*.json" --out all_models.html
```

The script expands the pattern itself, so the same command works in Windows
`cmd`, PowerShell and Unix shells. The page opens on Gemma-3-1B-it, the paper's
primary model; switching models resets the dials and returns to dual mode.

**One model per page**, or any selection:

```
python bake_interface.py --template index.html --atlas atlases/atlas_g31bit.json --out gemma-3-1b-it.html
python bake_interface.py --template index.html --atlas atlases/atlas_g31bit.json atlases/atlas_<code>.json --out two_models.html
```

`index.html` is the template. Opened directly it tries to fetch an atlas, which
browsers block for local files, so bake a page rather than opening the template.

To build an atlas for another model, from the reproduction pipeline:

```
python make_atlas.py --model <hf-id> --vecs <model>_vecs.npz --labels labels.csv \
    --probes canonical_probes.json --layer <L> --c-star <c*> --ekman ekman.json --out atlas.json
```

## Related

- [Verify](https://github.com/Sinerror/Human-to-LLM-interface-verify) — every
  number in the paper, recomputed on a laptop.
- [Reproduce](https://github.com/Sinerror/Human-to-LLM-interface-reproduce) — the
  full pipeline, from prompts to atlases.

## Licence

Code MIT. Word lists and atlases CC BY-SA 4.0 (see `LICENCE`).

The atlases hold int8 token projections derived from each model's unembedding
matrix, so each model's own licence may apply to them. Check before
redistributing: Gemma models are under the Gemma Terms of Use, Llama 3 under the
Meta Llama 3 Community License (which asks for "Built with Meta Llama 3"
attribution and a copy of the licence), Qwen2.5-1.5B, Mistral-7B-v0.3,
Pythia-6.9B and Mamba-2.8B under Apache 2.0, and Phi-2 under MIT — as stated on
their model cards at the time of writing.
