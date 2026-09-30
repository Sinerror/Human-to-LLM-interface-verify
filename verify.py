#!/usr/bin/env python3
"""
verify.py — recompute every number in the paper and compare against what it says.

  python verify.py                    # everything except the second encoder
  python verify.py --full             # also the second encoder (slow on CPU)
  python verify.py --only geometry    # one or more sections
  python verify.py --list             # the sections
  python verify.py --cache .embcache  # keep embeddings between runs

No GPU. Nothing here calls out to the analysis scripts: each check is
implemented in this file so a reader can audit one file rather than ten. Where a
check ports an analysis script, the docstring names the script, and the port
reproduces its choices (text condition, normalisation, row convention) even
where they differ from verify's own, because the paper quotes the script.

WHAT THIS VERIFIES, AND WHAT IT CANNOT

  Verified: the geometry of the basis, the axis validation, the steering
  measurements, simultaneous control, the dual-basis comparison, the
  superposition test, the linear baseline, and the interface readouts —
  everything computed from the shipped vectors, generations and atlas.

  Not verified: the extraction that produced the vectors and the generation that
  produced the texts, both of which need model weights and a GPU. The coherence
  edges are checked for internal consistency against the recorded ladders, not
  recomputed. The full reproduction archive covers those.

  A failing row is not necessarily an error in the paper. Tolerances are stated
  in expected.json, and a fail means the recomputed value fell outside one —
  which could be a library difference, a corrupted download, or a real
  discrepancy. The diagnostics printed alongside are there to tell those apart.

ROW CONVENTIONS — two are in use, and each check says which

  Per-trial and per-triple transfer matrices (sections control, magnitude) are
  fitted as in transfer_independent.py and transfer_matrix.py: least squares,
  then transposed, so ROWS ARE CONSTRUCTS and columns are dials. "Diagonal is
  the row maximum" there asks whether each construct is moved most by its own
  dial. The superposition matrices (section superposition) follow
  superposition_test.py: ROWS ARE DIALS. Both are reproduced as published.
"""
import argparse, csv, gzip, hashlib, itertools, json, os, re, sys, time
from collections import defaultdict
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SEED = 0
# Data keys. The paper calls the last axis isolation-belonging; the data keeps
# its original key, social_inner_outer. Same word lists, same direction.
AXES = ["valence", "heat", "arousal", "intensity", "antagonism_peace",
        "body_mind", "social_inner_outer"]
ALT_SEVEN = ["valence", "drive", "arousal", "heat", "security_alarm", "desire",
             "antagonism_peace"]
PRIMARY_MODEL = "google_gemma-3-1b-it"
MPNET = "sentence-transformers/all-mpnet-base-v2"
BGE = "BAAI/bge-large-en-v1.5"

SECTIONS = ["geometry", "nulls", "validation", "edges", "sharedwords",
            "interface", "linear", "geompred", "steering", "robustness",
            "crossboot", "control", "magnitude", "spread", "superposition",
            "commonmag"]
FULL_ONLY = ["encoder2"]
NEEDS_ENCODER = {"linear", "geompred", "steering", "robustness", "crossboot",
                 "control", "magnitude", "spread", "superposition", "encoder2"}
NEEDS_GENERATIONS = {"steering", "robustness", "crossboot", "control",
                     "magnitude", "spread", "superposition", "encoder2"}


# ─────────────────────────────────────────────── reporting
class Report:
    def __init__(self, expected):
        self.exp = expected
        self.rows = []
        self.dump = {}

    def spec(self, section, key):
        parts = key if isinstance(key, (tuple, list)) else key.split(".")
        spec = self.exp.get(section, {})
        for part in parts:
            spec = spec.get(part, {}) if isinstance(spec, dict) else {}
        return spec if isinstance(spec, dict) else {}

    def check(self, section, key, got, label=None):
        """`key` may be a string or a tuple of path components.

        Splitting a string on "." breaks on model names that contain one:
        "diag_is_max_by_model.mistralai_Mistral-7B-v0.3" splits into three
        parts and the lookup silently misses. Pass a tuple when any component
        may contain a dot.

        A spec may give `tol` (absolute), `rtol` (relative, for p-values whose
        scale spans orders of magnitude), or `at_most` (the paper states a
        bound - "changes by at most 0.001" - and any value under it agrees).
        """
        spec = self.spec(section, key)
        lab = label or (key if isinstance(key, str) else ".".join(key))
        if got is None or (isinstance(got, float) and not np.isfinite(got)):
            self.rows.append((lab, spec.get("value"), None, spec.get("tol"),
                              "NOT COMPUTED"))
            return
        if "at_most" in spec:          # the paper states a bound, not a value
            ok = got <= spec["at_most"] + 1e-12
            self.rows.append((lab, spec["at_most"], got, "<=", "PASS" if ok else "FAIL"))
            return
        if "value" not in spec:
            self.rows.append((lab, None, got, None, "NO EXPECTED"))
            return
        want = spec["value"]
        if "rtol" in spec:
            tol = spec["rtol"] * abs(want)
            ok = abs(got - want) <= tol * (1 + 1e-9)
            shown = f"{spec['rtol']:g}r"
        else:
            tol = spec.get("tol", 0)
            ok = abs(got - want) <= tol + 1e-12
            shown = tol
        self.rows.append((lab, want, got, shown, "PASS" if ok else "FAIL"))

    def check_list(self, section, key, got, label):
        """Ordered list, e.g. a top-ten token readout: counts matching positions."""
        spec = self.spec(section, key)
        want = spec.get("value")
        if want is None:
            self.rows.append((label, None, len(got), None, "NO EXPECTED"))
            self.note(f"    recomputed: {', '.join(map(str, got))}")
            return
        hit = sum(1 for a_, b_ in zip(want, got) if a_ == b_)
        ok = hit == len(want) and len(got) >= len(want)
        self.rows.append((label, len(want), hit, 0, "PASS" if ok else "FAIL"))
        if not ok:
            self.note(f"    paper:      {', '.join(map(str, want))}")
            self.note(f"    recomputed: {', '.join(map(str, got[:len(want)]))}")

    def compare(self, label, want, got, tol):
        """A check whose expectation lives in a structured spec rather than a
        {value, tol} leaf (the interface requests)."""
        if got is None:
            self.rows.append((label, want, None, tol, "NOT COMPUTED"))
            return
        ok = abs(got - want) <= tol + 1e-12
        self.rows.append((label, want, got, tol, "PASS" if ok else "FAIL"))

    def note(self, text):
        self.rows.append((text, None, None, None, "note"))

    def table(self):
        w = min(max([len(r[0]) for r in self.rows if r[4] != "note"] + [20]) + 2, 58)
        print(f"\n  {'CLAIM'.ljust(w)}{'PAPER':>10}{'RECOMPUTED':>13}"
              f"{'TOL':>8}   STATUS")
        print("  " + "-" * (w + 45))

        def fmt(x):
            if x is None:
                return "—"
            if isinstance(x, (bool, np.bool_)):
                return str(bool(x))
            if isinstance(x, (int, np.integer)):
                return str(int(x))
            x = float(x)
            if x != 0 and abs(x) < 0.01:
                return f"{x:.2e}"
            return f"{x:.3f}"
        for lab, want, got, tol, st in self.rows:
            if st == "note":
                print(f"  {lab}")
                continue
            ts = "—" if tol is None else (tol if isinstance(tol, str) else f"{tol:g}")
            print(f"  {lab[:w].ljust(w)}{fmt(want):>10}{fmt(got):>13}{ts:>8}   {st}")
        bad = [r for r in self.rows if r[4] == "FAIL"]
        miss = [r for r in self.rows if r[4] == "NO EXPECTED"]
        nc = [r for r in self.rows if r[4] == "NOT COMPUTED"]
        n = len([r for r in self.rows if r[4] in ("PASS", "FAIL")])
        print(f"\n  {n - len(bad)}/{n} checks passed"
              + (f", {len(miss)} with no expected value" if miss else "")
              + (f", {len(nc)} not computed" if nc else ""))
        return len(bad)


# ─────────────────────────────────────────────── embedding
class Embedder:
    """One embedding per unique text, computed once and looked up afterwards.

    Keyed on the FULL text. An earlier cache keyed on the first three texts of
    a list plus its length; two different lists can share both (the high sets
    of two axes of one triple begin with the same cell), and the second lookup
    silently returned the first axis's embedding. That turned a cross-alignment
    diagonal of 7/7 into 5/7 and looked like a finding about heat and arousal.
    A dictionary keyed on the string itself compares whole strings and cannot
    collide.

    Optional disk cache (--cache): sha256 of each text, per encoder. It makes
    reruns fast and changes nothing else; delete the folder to recompute.
    """

    def __init__(self, name, cache_dir=None):
        from sentence_transformers import SentenceTransformer
        self.name = name
        self.st = SentenceTransformer(name)
        self.vec = {}
        self.cache_path = None
        self.disk = {}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            slug = re.sub(r"[^A-Za-z0-9._-]", "_", name)
            self.cache_path = os.path.join(cache_dir, f"{slug}.npz")
            if os.path.exists(self.cache_path):
                z = np.load(self.cache_path, allow_pickle=False)
                self.disk = {str(k): v for k, v in zip(z["keys"], z["vecs"])}
                print(f"  embedding cache: {len(self.disk)} vectors from "
                      f"{self.cache_path}")

    @staticmethod
    def _sha(t):
        return hashlib.sha256(t.encode("utf-8", "replace")).hexdigest()

    def prime(self, texts):
        need = []
        seen = set()
        for t in texts:
            if t in self.vec or t in seen:
                continue
            if self.disk:
                v = self.disk.get(self._sha(t))
                if v is not None:
                    self.vec[t] = v
                    continue
            seen.add(t)
            need.append(t)
        if not need:
            return
        t0 = time.time()
        print(f"  embedding {len(need)} texts with {self.name} ...", flush=True)
        E = self.st.encode(need, convert_to_numpy=True,
                           show_progress_bar=len(need) > 5000, batch_size=64)
        for t, v in zip(need, E):
            self.vec[t] = v
        print(f"    {time.time() - t0:.0f} s", flush=True)
        if self.cache_path:
            for t, v in zip(need, E):
                self.disk[self._sha(t)] = v
            ks = list(self.disk)
            np.savez(self.cache_path, keys=np.array(ks),
                     vecs=np.array([self.disk[k] for k in ks]))

    def __call__(self, texts):
        texts = list(texts)
        missing = [t for t in texts if t not in self.vec]
        if missing:
            self.prime(missing)
        return np.array([self.vec[t] for t in texts], dtype=np.float64)

    def contrast(self, pos, neg):
        return unit(self(pos).mean(0) - self(neg).mean(0))


# ─────────────────────────────────────────────── shared
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# A key present only in the superseded label file. The correction renamed
# "self-contempt::fallback" to "self-contempt::self-contempt", along with six
# other glosses, so its presence identifies the old file exactly.
SUPERSEDED_MARKER = "self-contempt::fallback"


def load_labels(p, strict=True):
    """Labels, with a check that this is the corrected file.

    Using the superseded file shifts the axis directions slightly and every
    geometry figure with them - 0.177 rather than 0.187 for the mean Gram
    off-diagonal, and so on. Nothing errors; the numbers are simply wrong.
    """
    out = {}
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        out[r["key"]] = (r.get("final", "").strip().upper() or r["heuristic"])
    if strict and SUPERSEDED_MARKER in out:
        print(f"\n  !! {os.path.basename(p)} contains '{SUPERSEDED_MARKER}', "
              f"a key that only the SUPERSEDED label file has.")
        print("     Every geometry figure will be wrong without raising an "
              "error. Expect the Gram rows to fail.\n")
    return out


def wordvecs(path, labels, layer=None, emo_only=True, center_with=None):
    """Vectors at one layer, from either a single-layer or a full file.

    A full extraction works too, but then the layer has to be chosen: taking
    index 0 would silently read the embedding layer. An ambiguous file without
    a layer argument is therefore an error rather than a guess.
    """
    z = np.load(path, allow_pickle=False)
    keys = [str(k) for k in z["keys"]]
    words = np.array([str(w).lower() for w in z["words"]])
    X = z["vecs"].astype(np.float64)
    if X.ndim == 3:
        layers = [int(v) for v in np.asarray(z["layers"])] \
            if "layers" in z else list(range(X.shape[1]))
        if X.shape[1] == 1:
            idx = 0
        elif layer is not None and layer in layers:
            idx = layers.index(layer)
        else:
            raise ValueError(
                f"{os.path.basename(path)} holds {X.shape[1]} layers "
                f"{layers[:3]}...{layers[-1]}; pass the selected layer.")
        X = X[:, idx, :]
    m = (np.array([labels.get(k) == "EMO" for k in keys]) if emo_only
         else np.ones(len(keys), bool))
    Xc = X[m] - (X[m].mean(0) if center_with is None else center_with)
    acc = defaultdict(list)
    for w, v in zip(words[m], Xc):
        acc[w].append(v)
    return {w: np.mean(vs, 0) for w, vs in acc.items()}


def axes_from(wv, probes, names=AXES):
    P, kept = [], []
    for nm in names:
        s = probes.get(nm, {})
        pos = [w.lower() for w in s.get("pos", []) if w.lower() in wv]
        neg = [w.lower() for w in s.get("neg", []) if w.lower() in wv]
        if len(pos) >= 4 and len(neg) >= 4:
            P.append(unit(np.mean([wv[w] for w in pos], 0)
                          - np.mean([wv[w] for w in neg], 0)))
            kept.append(nm)
    return np.array(P), kept


def vec_path(model):
    return os.path.join(DATA, "vectors_layer", f"{model}.npz")


def model_gram(model, probes, labels, layers):
    """Activation Gram of the seven axes for one model, or None."""
    p = vec_path(model)
    if not os.path.exists(p):
        return None
    P, names = axes_from(wordvecs(p, labels, layers.get(model)), probes)
    return P @ P.T if names == AXES else None


def slug_for(bare, slugs):
    """Generation files carry the bare model name (gemma-3-1b-it); vectors and
    expected.json carry the hub slug (google_gemma-3-1b-it). Match on suffix,
    never by splitting on '.' - model names contain dots."""
    hits = [s for s in slugs if s == bare or s.endswith("_" + bare)]
    return hits[0] if len(hits) == 1 else None


def sep7(M):
    """Separability of a pooled matrix, ROWS = DIAL: 1 / mean |off-diagonal|
    after dividing each row by its own diagonal (as make_figures.fig7)."""
    Mn = M / np.abs(np.diag(M))[:, None]
    return 1.0 / np.abs(Mn[~np.eye(len(M), dtype=bool)]).mean()


def anti7(M):
    """Antisymmetric share of cross-talk on a pooled matrix, scale-free:
    Mn = M / sqrt(|diag| x |diag|), then antisym / (sym + antisym)."""
    k = len(M)
    iu = np.triu_indices(k, 1)
    d = np.abs(np.diag(M))
    Mn = M / np.sqrt(np.outer(d, d))
    sy = np.abs(((Mn + Mn.T) / 2)[iu]).mean()
    an = np.abs(((Mn - Mn.T) / 2)[iu]).mean()
    return an / (sy + an)


def exact_spearman_p(x, y):
    """Exact two-sided p for Spearman's rho by enumerating all rankings.
    Small n only (7 axes -> 5040 permutations)."""
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    def rho(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float(a @ b / np.sqrt((a @ a) * (b @ b)))
    r0 = rho(rx, ry)
    hits = tot = 0
    for p in itertools.permutations(ry):
        tot += 1
        hits += abs(rho(rx, np.array(p))) >= abs(r0) - 1e-12
    return r0, hits / tot


# ─────────────────────────────────────────────── vector-only checks
def check_geometry(rep, probes, labels, layers):
    """Sections 3.4, 5.1-5.3: the Gram matrix and what follows from it."""
    print("\n[geometry] basis geometry - vectors only")
    wv = wordvecs(vec_path(PRIMARY_MODEL), labels, layers.get(PRIMARY_MODEL))
    P, names = axes_from(wv, probes)
    G = P @ P.T
    k = len(P)
    iu = np.triu_indices(k, 1)
    Gi = np.linalg.inv(G)
    iso = np.sqrt(np.diag(Gi))
    ix = {n: i for i, n in enumerate(names)}
    S = "basis_geometry"
    rep.check(S, "n_axes", len(names), "axes built")
    rep.check(S, "gram_mean_abs_offdiag", float(np.abs(G[iu]).mean()),
              "Gram mean |cos|")
    rep.check(S, "gram_max_abs_offdiag", float(np.abs(G[iu]).max()),
              "Gram max |cos|")
    rep.check(S, "gram_condition", float(np.linalg.cond(G)), "Gram condition number")
    if len(names) == 7:
        a_ = ix["antagonism_peace"]
        rep.check(S, "gram_intensity_antagonism", float(G[ix["intensity"], a_]),
                  "  Gram intensity-antagonism")
        rep.check(S, "gram_valence_antagonism", float(G[ix["valence"], a_]),
                  "  Gram valence-antagonism")
        rep.check(S, "gram_heat_antagonism", float(G[ix["heat"], a_]),
                  "  Gram heat-antagonism")
        leak = {n: float(np.sqrt(sum(G[j, i] ** 2 for j in range(k) if j != i)))
                for i, n in enumerate(names)}
        worst = max(leak, key=leak.get)
        rep.check(S, "leak_max", leak[worst], f"leak per unit, max ({worst})")
    rep.check(S, "iso_cost_min", float(iso.min()), "iso-cost min")
    rep.check(S, "iso_cost_max", float(iso.max()), "iso-cost max")
    lam = np.linalg.eigvalsh(G)
    rep.check(S, "request_cost_min", float(1 / np.sqrt(lam.max())),
              "request cost, min (1/sqrt lambda_max)")
    rep.check(S, "request_cost_max", float(1 / np.sqrt(lam.min())),
              "request cost, max (1/sqrt lambda_min)")
    # section 5.3, Gram-Schmidt in the order AXES lists them
    Q, _ = np.linalg.qr(P.T)
    for i, n in enumerate(names):
        rep.check(S, ("gram_schmidt", n), float(abs(P[i] @ Q[:, i])),
                  f"  Gram-Schmidt cos, {n}")
    # section 3.4, the alternative seven
    Pa, na = axes_from(wv, probes, ALT_SEVEN)
    if len(na) == 7:
        Ga = Pa @ Pa.T
        iua = np.triu_indices(7, 1)
        rep.check(S, "alt_seven_mean_abs_offdiag", float(np.abs(Ga[iua]).mean()),
                  "alternative seven, mean |cos|")
        rep.check(S, "alt_seven_condition", float(np.linalg.cond(Ga)),
                  "alternative seven, condition")
        rep.check(S, "alt_seven_drive_desire",
                  float(abs(Ga[na.index("drive"), na.index("desire")])),
                  "alternative seven, drive-desire |cos|")
    else:
        rep.note(f"  alternative seven: only {len(na)} axes buildable; skipped")
    rep.dump["gram"] = dict(names=names, G=G.tolist())
    return dict(G=G, names=names, wv=wv)


def check_nulls(rep, probes, labels, layers):
    """Section 3.3: the two nulls, which are CROSS-MODEL rank correlations of
    scores over a shared vocabulary, not cosines within one model."""
    from scipy.stats import spearmanr
    print("[nulls] null construction - cross-model rank correlation")
    va, vb = vec_path(PRIMARY_MODEL), vec_path("google_gemma-3-1b-pt")
    if not os.path.exists(vb):
        rep.note("  second model absent; null check needs two models, skipped")
        return
    A = wordvecs(va, labels, layers.get(PRIMARY_MODEL))
    B = wordvecs(vb, labels, layers.get("google_gemma-3-1b-pt"))
    shared = sorted(set(A) & set(B))
    dim = len(next(iter(A.values())))
    rng = np.random.default_rng(SEED)
    rv = []
    for _ in range(200):
        r1, r2 = unit(rng.normal(size=dim)), unit(rng.normal(size=dim))
        s1 = np.array([A[w] @ r1 for w in shared])
        s2 = np.array([B[w] @ r2 for w in shared])
        rv.append(abs(float(spearmanr(s1, s2).statistic)))

    def contrast(wv, pos, neg):
        return unit(np.mean([wv[w] for w in pos], 0)
                    - np.mean([wv[w] for w in neg], 0))
    wc = []
    for _ in range(200):
        s = list(rng.choice(shared, 16, replace=False))
        d1, d2 = contrast(A, s[:8], s[8:]), contrast(B, s[:8], s[8:])
        x1 = np.array([A[w] @ d1 for w in shared])
        x2 = np.array([B[w] @ d2 for w in shared])
        wc.append(abs(float(spearmanr(x1, x2).statistic)))
    rep.check("axis_validation", "random_vector_null_p95",
              float(np.percentile(rv, 95)), "random-vector null p95")
    rep.check("axis_validation", "word_contrast_null_p95",
              float(np.percentile(wc, 95)), "word-contrast null p95")


def _lowo(wv, pos, neg):
    """Hold out one word per pole, rebuild from the rest, check the ordering."""
    c = t = 0
    for wp in pos:
        for wn in neg:
            d = unit(np.mean([wv[w] for w in pos if w != wp], 0)
                     - np.mean([wv[w] for w in neg if w != wn], 0))
            c += int(float(wv[wp] @ d) > float(wv[wn] @ d))
            t += 1
    return c / t


def _auc(a, b):
    x = np.concatenate([a, b])
    order = x.argsort().argsort() + 1.0
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, order)
    order = (sums / cnt)[inv]
    n1, n0 = len(a), len(b)
    return float((order[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _mu(path, labels, layer):
    z = np.load(path, allow_pickle=False)
    keys = [str(k) for k in z["keys"]]
    X = z["vecs"].astype(np.float64)
    if X.ndim == 3:
        ls = [int(v) for v in np.asarray(z["layers"])] if "layers" in z else [0]
        X = X[:, ls.index(layer) if layer in ls else 0, :]
    m = np.array([labels.get(k) == "EMO" for k in keys])
    return X[m].mean(0, keepdims=True)


def check_validation(rep, probes, labels, layers):
    """Section 3.3: leave-one-word-out and pre-registered unseen words."""
    import glob as _g
    print("[validation] held-out words and pre-registered vocabulary")
    wv = wordvecs(vec_path(PRIMARY_MODEL), labels, layers.get(PRIMARY_MODEL))
    rng = np.random.default_rng(SEED)
    per = {}
    for nm in AXES:
        spec = probes.get(nm, {})
        pos = [w.lower() for w in spec.get("pos", []) if w.lower() in wv]
        neg = [w.lower() for w in spec.get("neg", []) if w.lower() in wv]
        if len(pos) < 4 or len(neg) < 4:
            continue
        acc = _lowo(wv, pos, neg)
        pool = pos + neg
        nl = []
        for _ in range(60):
            sh = list(rng.permutation(pool))
            nl.append(_lowo(wv, sh[:len(pos)], sh[len(pos):]))
        per[nm] = (acc, float(np.percentile(nl, 95)))
    if per:
        accs = [v[0] for v in per.values()]
        nulls = [v[1] for v in per.values()]
        S = "axis_validation"
        rep.check(S, "lowo_accuracy_min", min(accs), "LOWO accuracy, min")
        rep.check(S, "lowo_accuracy_max", max(accs), "LOWO accuracy, max")
        rep.check(S, "axes_lowo_perfect", sum(a_ >= 1 - 1e-12 for a_ in accs),
                  "axes at LOWO accuracy 1.000")
        rep.check(S, "lowo_null_min", min(nulls), "LOWO permuted null, min")
        rep.check(S, "lowo_null_max", max(nulls), "LOWO permuted null, max")
        rep.check(S, "axes_clearing_lowo", sum(a_ > n_ for a_, n_ in per.values()),
                  "axes clearing the LOWO null")
        if "body_mind" in per:
            rep.check(S, "lowo_body_mind_accuracy", per["body_mind"][0],
                      "  LOWO body-mind accuracy")
            rep.check(S, "lowo_body_mind_null", per["body_mind"][1],
                      "  LOWO body-mind null")

    pred_p = os.path.join(DATA, "predictions.csv")
    hd = os.path.join(DATA, "heldout_vectors")
    if not os.path.exists(pred_p) or not os.path.isdir(hd):
        rep.note("  predictions.csv or heldout_vectors/ absent; unseen-word check skipped")
        return
    avail = sorted(_g.glob(os.path.join(hd, "*.npz")))

    def held_path(model):
        exact = os.path.join(hd, f"{model}.npz")
        if os.path.exists(exact):
            return exact
        hits = [f for f in avail if model in os.path.basename(f)]
        return hits[0] if len(hits) == 1 else None
    preds = defaultdict(list)
    for r in csv.DictReader(open(pred_p, newline="", encoding="utf-8")):
        preds[r["probe"]].append((r["word"].strip().lower(),
                                  r["side"].strip().lower()))
    aucs, clears, total, skipped = [], 0, 0, []
    for model, layer in sorted(layers.items()):
        hv, mv = held_path(model), vec_path(model)
        if hv is None or not os.path.exists(mv):
            skipped.append(model)
            continue
        base = wordvecs(mv, labels, layer)
        # held-out words are centred on the ORIGINAL corpus mean, so that the
        # axis and the words share a frame
        held = wordvecs(hv, labels, layer, emo_only=False,
                        center_with=_mu(mv, labels, layer))
        for nm in AXES:
            spec = probes.get(nm, {})
            pos = [w.lower() for w in spec.get("pos", []) if w.lower() in base]
            neg = [w.lower() for w in spec.get("neg", []) if w.lower() in base]
            if len(pos) < 4 or len(neg) < 4 or nm not in preds:
                continue
            d = unit(np.mean([base[w] for w in pos], 0)
                     - np.mean([base[w] for w in neg], 0))
            hp = [w for w, sd in preds[nm] if sd == "pos" and w in held]
            hn = [w for w, sd in preds[nm] if sd == "neg" and w in held]
            if len(hp) < 3 or len(hn) < 3:
                continue
            sp = np.array([float(held[w] @ d) for w in hp])
            sn = np.array([float(held[w] @ d) for w in hn])
            a_ = _auc(sp, sn)
            pool = [w for w in held if w not in set(hp) | set(hn)]
            nl = []
            for _ in range(200):
                s_ = rng.choice(pool, len(hp) + len(hn), replace=False)
                nl.append(_auc(np.array([float(held[w] @ d) for w in s_[:len(hp)]]),
                               np.array([float(held[w] @ d) for w in s_[len(hp):]])))
            aucs.append(a_)
            clears += int(a_ > np.percentile(nl, 95))
            total += 1
    if skipped:
        rep.note(f"  {len(skipped)} model(s) skipped for unseen words: {skipped}")
    if aucs:
        rep.check("axis_validation", "unseen_word_auc_min", min(aucs),
                  "pre-registered unseen-word AUC, min")
        n_models = max(1, total // 7)
        rep.check("axis_validation", "axes_clearing_unseen_all_models",
                  7 if clears == total else clears // n_models,
                  f"axes clearing in all {n_models} models")



def check_commonmag(rep):
    """Sections 3.5 and 7.2: the same simultaneous-control experiment run at one
    common magnitude (c = 0.08 for every model) reads 54% rather than 86%.

    That run predates the coherence-edge protocol and its generations are in the
    reproduction archive, not here; what ships is its fitted per-repetition
    transfer matrices. This check recomputes the headline rate from them with the
    same rule as the main result: each row divided by its largest absolute entry,
    the diagonal counted when it is that row's maximum.
    """
    print("[commonmag] simultaneous control at one common magnitude (archived run)")
    p = os.path.join(DATA, "common_magnitude_transfer.json")
    if not os.path.exists(p):
        rep.note("common_magnitude_transfer.json absent; skipped")
        return
    d = json.load(open(p, encoding="utf-8"))
    rates = []
    for model, v in d.items():
        a = b = 0
        for t in v["per_triple"]:
            M = np.array(t["M"], float)
            Mn = M / (np.abs(M).max(axis=1, keepdims=True) + 1e-12)
            a += sum(int(np.argmax(np.abs(Mn[i])) == i) for i in range(3))
            b += 3
        rates.append(a / b)
    rep.check("common_magnitude", "diag_dominant_mean", float(np.mean(rates)),
              "diag-is-max at one common magnitude, mean")
    rep.check("common_magnitude", "n_models", len(rates), "  models in the archived run")


def check_edges(rep):
    """Section 3.5: c* recomputed from each stored ladder by its stored rule."""
    print("[edges] coherence edges - consistency against the recorded ladders")
    p = os.path.join(DATA, "edges.json")
    if not os.path.exists(p):
        rep.note("edges.json absent; skipped")
        return
    E = json.load(open(p, encoding="utf-8"))
    exp = rep.exp["coherence_edges"]["c_star"]
    bad = 0
    for model, cs in exp.items():
        rec = E.get(model)
        if rec is None:
            bad += 1
            continue
        p0 = rec["baseline_pass_rate"]
        need = max(rec["criterion"]["retention"] * p0, rec["criterion"]["min_abs"])
        got = None
        for L in rec["levels"]:
            if L["c"] == 0.0:
                continue
            if L["pass_rate"] >= need:
                got = L["c"]
            else:
                break
        if got != cs:
            bad += 1
            rep.note(f"    {model}: recorded c*={cs}, ladder implies {got}")
    vals = list(exp.values())
    rep.check("coherence_edges", "spread_ratio", float(max(vals) / min(vals)),
              "c* spread, max/min")
    rep.check("coherence_edges", "ladders_disagreeing", bad,
              "c* not reproduced by its ladder")


def check_shared_words(rep, probes, labels, layers):
    """Section 5.1: how much of the Gram is words shared between lists.
    Port of the shared-word block of derive_paper_numbers.py, including its
    random-removal null and its seed."""
    print("[sharedwords] words shared between lists")
    wv = wordvecs(vec_path(PRIMARY_MODEL), labels, layers.get(PRIMARY_MODEL))
    poles = {}
    for nm in AXES:
        s = probes[nm]
        poles[nm] = ([w.lower() for w in s["pos"] if w.lower() in wv],
                     [w.lower() for w in s["neg"] if w.lower() in wv])
    cnt = defaultdict(int)
    for nm in AXES:
        for side in ("pos", "neg"):
            for w in probes[nm][side]:
                cnt[w.lower()] += 1
    shared = sorted(w for w, c in cnt.items() if c > 1)

    def gram_of(lists):
        Q = np.array([unit(np.mean([wv[w] for w in ps], 0)
                           - np.mean([wv[w] for w in ng], 0)) for ps, ng in lists])
        return Q @ Q.T
    full = [poles[nm] for nm in AXES]
    nos = [([w for w in ps if w not in shared], [w for w in ng if w not in shared])
           for ps, ng in full]
    G, G_ns = gram_of(full), gram_of(nos)
    iu = np.triu_indices(7, 1)
    rng2 = np.random.default_rng(0)
    null = []
    for _ in range(500):
        lists = []
        for (ps, ng), (ps2, ng2) in zip(full, nos):
            pair = []
            for pole, kept in ((ps, ps2), (ng, ng2)):
                k = len(pole) - len(kept)
                cand = [w for w in pole if w not in shared]
                drop = set(rng2.choice(cand, min(k, max(len(cand) - 2, 0)),
                                       replace=False)) if k else set()
                pair.append([w for w in pole if w not in drop])
            lists.append(tuple(pair))
        null.append(float(np.abs(gram_of(lists)[iu]).mean()))
    null = np.array(null)
    m_ns = float(np.abs(G_ns[iu]).mean())
    ia, ii = AXES.index("antagonism_peace"), AXES.index("intensity")
    S = "shared_words"
    rep.check(S, "n_shared", len(shared), "words in more than one list")
    rep.check(S, "mean_without", m_ns, "Gram mean |cos| without them")
    rep.check(S, "condition_without", float(np.linalg.cond(G_ns)),
              "  condition without them")
    rep.check(S, "null_mean", float(null.mean()), "random-removal null, mean")
    rep.check(S, "null_p025", float(np.percentile(null, 2.5)), "  null 2.5%")
    rep.check(S, "null_p975", float(np.percentile(null, 97.5)), "  null 97.5%")
    rep.check(S, "null_draws_as_low", int((null <= m_ns).sum()),
              "  null draws as low (of 500)")
    rep.check(S, "intensity_antagonism_full", float(G[ii, ia]),
              "intensity-antagonism, full lists")
    rep.check(S, "intensity_antagonism_without", float(G_ns[ii, ia]),
              "intensity-antagonism, without shared")
    for a_, b_ in (("heat", "intensity"), ("valence", "intensity")):
        i, j = AXES.index(a_), AXES.index(b_)
        rep.note(f"    {a_}-{b_}: {G[i, j]:+.3f} -> {G_ns[i, j]:+.3f} "
                 f"(paper: 'fall to near zero')")
    rep.dump["shared_words"] = shared


def load_atlas():
    for fn in ("atlas_g31bit.json.gz", "atlas_g31bit.json"):
        p = os.path.join(DATA, fn)
        if os.path.exists(p):
            op = gzip.open if fn.endswith(".gz") else open
            with op(p, "rt", encoding="utf-8") as f:
                return json.load(f)
    return None


def interface_readout(A, levels, mode):
    """Exactly what interface/index.html computes, including its float32
    accumulation and its stable descending sort, so ties break the same way."""
    names = A["axes"]
    G = np.array(A["gram"], float)
    Gi = np.linalg.inv(G)
    lv = np.array([float(levels.get(n, 0.0)) for n in names])
    c = lv.copy() if mode == "dual" else G @ lv
    w = Gi @ c
    cu = unit(w)
    scale = np.array(A["scale"], float)
    s = np.zeros(len(A["tokens"]), np.float32)
    for j in range(len(names)):
        wj = cu[j] * scale[j]
        if not wj:
            continue
        col = np.asarray(A["coords"][j], np.float64)
        s = (s.astype(np.float64) + col * wj).astype(np.float32)
    order = np.argsort(-s.astype(np.float64), kind="stable")
    tokens = [A["tokens"][i] for i in order[:10]]
    gdot = lambda x, y: float(x @ (Gi @ y))
    cc = gdot(c, c)
    ek = sorted(((nm, gdot(np.array(v, float), c)
                  / np.sqrt(max(gdot(np.array(v, float), np.array(v, float)) * cc,
                                1e-12)))
                 for nm, v in (A.get("ekman") or {}).items()),
                key=lambda t: -t[1])
    cost = float(np.sqrt(max(lv @ Gi @ lv, 0)) / max(np.linalg.norm(lv), 1e-9))
    return dict(cursor=dict(zip(names, c.tolist())), tokens=tokens, ekman=ek,
                cost=cost)


def check_interface(rep, geo):
    """Section 4: the two requests in the interface table, recomputed from the
    shipped atlas with the page's own arithmetic."""
    print("[interface] the control surface's readouts, from the shipped atlas")
    A = load_atlas()
    if A is None:
        rep.note("  data/atlas_g31bit.json absent; interface checks skipped")
        return
    G = np.array(A["gram"], float)
    if geo is not None and len(geo["names"]) == len(A["axes"]):
        perm = [geo["names"].index(n) for n in A["axes"]]
        dg = float(np.abs(G - geo["G"][np.ix_(perm, perm)]).max())
        rep.check("interface", "atlas_gram_vs_vectors_max_diff", dg,
                  "atlas Gram = Gram from shipped vectors (max |diff|)")
    ispec = rep.exp.get("interface", {})
    tol = ispec.get("tol", 0.005)
    for n, req in enumerate(ispec.get("requests", []), 1):
        for mode in ("raw", "dual"):
            R = interface_readout(A, req["levels"], mode)
            e = req.get(mode, {})
            for ax_, v in (e.get("cursor") or {}).items():
                rep.compare(f"request {n} {mode}: {ax_} arrives at", v,
                            R["cursor"].get(ax_), tol)
            if "cost" in e:
                rep.compare(f"request {n} {mode}: cost vs orthogonal basis",
                            e["cost"], R["cost"], tol)
            if "tokens" in e:
                want = e["tokens"]
                hit = sum(1 for a_, b_ in zip(want, R["tokens"]) if a_ == b_)
                ok = hit == len(want)
                rep.rows.append((f"request {n} {mode}: top-ten tokens, in order",
                                 len(want), hit, 0, "PASS" if ok else "FAIL"))
                if not ok:
                    rep.note(f"      paper:      {', '.join(want)}")
                    rep.note(f"      recomputed: {', '.join(R['tokens'])}")
            names = [x[0] for x in R["ekman"]]
            for i, (nm, v) in enumerate(e.get("ekman") or []):
                got = dict(R["ekman"]).get(nm)
                rep.compare(f"request {n} {mode}: nearest Ekman #{i + 1} {nm}",
                            v, got, tol)
                if got is not None and names.index(nm) != i:
                    rep.note(f"      {nm} ranks {names.index(nm) + 1} of "
                             f"{len(names)}; the paper lists it {i + 1}")
            rep.dump.setdefault("interface", {})[f"r{n}_{mode}"] = R


def check_linear(rep, probes, labels, layers, emb, grams):
    """Port of linear_baseline.py: what a perfectly linear model would show in
    this pipeline, per model over all 35 triples. THREE-DIAL, per-trial
    statistics - not comparable with the pooled single-dial figures."""
    print("[linear] linear baseline - encoder overlap and geometry only")
    A = np.array([[emb.contrast(probes[i]["pos"], probes[i]["neg"])
                   @ emb.contrast(probes[j]["pos"], probes[j]["neg"])
                   for j in AXES] for i in AXES])
    iu = np.triu_indices(7, 1)
    S = "linear_baseline"
    rep.check(S, "encoder_gram_mean", float(np.abs(A[iu]).mean()),
              "encoder-probe Gram, mean |cos|")
    rep.check(S, "encoder_gram_max", float(np.abs(A[iu]).max()),
              "encoder-probe Gram, max |cos|")
    out = {}
    for m, G in grams.items():
        out[m] = trio_stats(*linear_predicted(G, A, True)[:2])
    if not out:
        rep.note("  no model Grams; skipped")
        return A
    g = lambda k: float(np.mean([v[k] for v in out.values() if np.isfinite(v[k])]))
    rep.check(S, "pred_symmetric_change", g("sym"),
              "linear prediction: symmetric off-diag change")
    rep.check(S, "pred_antisymmetric_change", g("anti"),
              "linear prediction: antisymmetric change")
    rep.check(S, "pred_separability_gain", g("sep"),
              "linear prediction: separability x (THREE-dial)")
    rep.check(S, "pred_antisymmetric_share", g("raw_anti_share"),
              "linear prediction: antisym share (per-trial)")
    rep.dump["linear_baseline"] = out
    return A


def linear_predicted(G, A, harness):
    """linear_baseline.predicted: per triple, the 3x3 main-effect matrices a
    linear model produces in the THREE-dial design, raw and dual, with the
    harness's per-cell rescaling. ROWS = CONSTRUCT (Y.T @ L8), as in the
    script. Also returns the triples, for pooling."""
    L8 = np.array(list(itertools.product([-1, 1], repeat=3)), float)
    Gi = np.linalg.inv(G)
    R, D, T = [], [], []
    for idx in itertools.combinations(range(7), 3):
        ix = np.ix_(idx, idx)
        G3, Gi3, A3 = G[ix], Gi[ix], A[ix]
        yr = np.array([A3 @ (G3 @ l) / (np.sqrt(l @ G3 @ l) if harness else 1)
                       for l in L8])
        yd = np.array([A3 @ l / (np.sqrt(l @ Gi3 @ l) if harness else 1)
                       for l in L8])
        R.append((yr.T @ L8) / 8.0); D.append((yd.T @ L8) / 8.0)
        T.append([AXES[i] for i in idx])
    return R, D, T


def pool7(mats, triples):
    """Average each ordered pair's entry over every triple containing it."""
    acc = defaultdict(list)
    for M, tri in zip(mats, triples):
        for i, x in enumerate(tri):
            for j, y in enumerate(tri):
                acc[(AXES.index(x), AXES.index(y))].append(M[i, j])
    return np.array([[np.mean(acc[(i, j)]) for j in range(7)] for i in range(7)])


def trio_stats(R, D):
    """linear_baseline.stats: component changes and per-triple separability on
    3x3 matrices, unnormalised for the components, row-max-normalised for the
    ratio."""
    iu = np.triu_indices(3, 1); il = np.tril_indices(3, -1)

    def parts(Ms):
        return (np.mean([np.abs(np.diag(X)).mean() for X in Ms]),
                np.mean([np.abs(((X + X.T) / 2)[iu]).mean() for X in Ms]),
                np.mean([np.abs(((X - X.T) / 2)[iu]).mean() for X in Ms]))

    def ratio(Ms):
        out = []
        for X in Ms:
            Xn = X / (np.abs(X).max(1, keepdims=True) + 1e-12)
            off = np.concatenate([np.abs(Xn[iu]), np.abs(Xn[il])]).mean()
            out.append(np.abs(np.diag(Xn)).mean() / max(off, 1e-9))
        return np.mean(out)
    a, b = parts(R), parts(D)
    ch = lambda i: (b[i] / a[i] - 1) if a[i] > 1e-9 else float("nan")
    rr, rd = ratio(R), ratio(D)
    return dict(diag=ch(0), sym=ch(1), anti=ch(2),
                sep=(rd / rr) if rr > 0 and np.isfinite(rd) and rd < 1e3 else float("inf"),
                raw_anti_share=a[2] / (a[1] + a[2]) if a[1] + a[2] > 0 else 0.0)


def check_geompred(rep, A, grams, ctx):
    """Section 7.3: what geometry alone predicts ONE DIAL AT A TIME.

    For a linear model with activation Gram G and encoder-probe overlap A, a
    single raw dial i moves activation coordinates by G e_i and the readout by
    A G e_i; a single dual dial moves them by e_i (up to its length) and the
    readout by A e_i. With ROWS = DIAL the pooled single-dial matrices are
    therefore (A G)^T = G A for raw and A for dual, and the predicted gain in
    separability is sep7(A) / sep7(G A). The per-row lengths of the dual
    directions scale rows and cancel in sep7.

    Two antisymmetric shares are also reported, each computed exactly as its
    observed counterpart: of G A (single dial, pooled, scale-free - the
    comparator for the observed single-dial share), and of the linear
    baseline's three-dial matrices pooled the same way (the comparator for the
    observed three-dial pooled share). The linear baseline's own per-trial
    share is a third quantity again, and is checked in section linear.
    """
    print("[geompred] single-dial geometric prediction, all models")
    rows = {}
    for m, G in sorted(grams.items()):
        Mr, Md = G @ A, A
        R3, _, T3 = linear_predicted(G, A, True)
        rows[m] = dict(pred_gain=float(sep7(Md) / sep7(Mr)),
                       anti_share_geometric=float(anti7(Mr)),
                       anti_share_geometric_three=float(anti7(pool7(R3, T3))))
    if not rows:
        rep.note("  no model Grams; skipped")
        return
    S = "geometric_prediction"
    if PRIMARY_MODEL in rows:
        rep.check(S, "primary_pred_gain", rows[PRIMARY_MODEL]["pred_gain"],
                  "Gemma-3-1B-it: predicted single-dial gain x")
    rep.check(S, "mean_pred_gain", float(np.mean([r["pred_gain"] for r in rows.values()])),
              f"mean over {len(rows)} models: predicted gain x")
    rep.check(S, "mean_anti_share_geometric",
              float(np.mean([r["anti_share_geometric"] for r in rows.values()])),
              "  geometric antisym share, single dial, pooled")
    rep.check(S, "mean_anti_share_geometric_three",
              float(np.mean([r["anti_share_geometric_three"] for r in rows.values()])),
              "  geometric antisym share, three dials, pooled")
    ctx["geompred"] = rows
    rep.dump["geometric_prediction"] = rows


# ─────────────────────────────────────────────── generations
def dedup(text, max_period=4):
    """Collapse immediately-repeating token cycles to a single copy.

    The paper's primary condition is `dedup`, not raw text: a mean-pooled
    encoder weights repetition, so a generation that repeats the steered word
    forty times registers alignment without any semantic change.
    """
    toks = re.findall(r"\S+", str(text))
    n = len(toks)
    if n < 2:
        return text
    low = [t.lower() for t in toks]
    out, i = [], 0
    while i < n:
        bp, br = 1, 1
        for per in range(1, max_period + 1):
            if i + per > n:
                break
            reps = 1
            while (i + per * (reps + 1) <= n
                   and low[i + per * reps: i + per * (reps + 1)]
                   == low[i: i + per]):
                reps += 1
            if reps > br:
                bp, br = per, reps
        out.extend(toks[i: i + bp])
        i += bp * br
    return " ".join(out)


def degenerate(text):
    """embed_align.degeneracy, reduced to its verdict: looped or collapsed.
    Evaluated on the RAW text, as embed_align does for its `coherent` condition."""
    from collections import Counter
    toks = re.findall(r"\S+", text)
    if not toks:
        return True
    uniq = len(set(t.lower() for t in toks)) / len(toks)
    punct = sum(1 for t in toks if not re.search(r"[A-Za-z0-9]", t)) / len(toks)
    run = best = 1
    for i in range(1, len(toks)):
        run = run + 1 if toks[i].lower() == toks[i - 1].lower() else 1
        best = max(best, run)
    low = [t.lower() for t in toks]
    bg = Counter(zip(low, low[1:]))
    top_bg = max(bg.values()) if bg else 0
    cyc = 1
    for per in range(1, 5):
        run = 1
        for i in range(per, len(low)):
            run = run + 1 if low[i] == low[i - per] else 1
            cyc = max(cyc, run // per if per > 1 else run)
    looped = best >= 4 or top_bg >= 3 or cyc >= 3 or uniq < 0.50
    collapsed = punct > 0.5 or len(toks) < 5
    return bool(looped or collapsed)


def load_generations(folder):
    """(model, arm, trial_id, triple) -> {levels-sign tuple: [raw texts]}.

    Files are <model>_<arm>_generations.jsonl[.gz]; the arm is stripped from
    the model name, or each arm registers as a separate model and signal and
    control never pair. Insertion order is file order then line order, which
    the ports below rely on where the original scripts did.
    """
    import glob
    paths = sorted(glob.glob(os.path.join(folder, "*.jsonl"))
                   + glob.glob(os.path.join(folder, "*.jsonl.gz")))
    cells = defaultdict(dict)
    stems = {}
    for path in paths:
        base = os.path.basename(path)
        base = base[: base.index("_generations.jsonl")]
        for suf in ("_raw", "_dual", "_control"):
            if base.endswith(suf):
                model, arm = base[: -len(suf)], suf[1:]
                break
        else:
            model, arm = base, "raw"
        opener = gzip.open if path.endswith(".gz") else open
        for line in opener(path, "rt", encoding="utf-8"):
            r = json.loads(line)
            if r.get("arm", arm) != arm:
                continue
            tri = tuple(r["triple"] if isinstance(r["triple"], list)
                        else json.loads(r["triple"]))
            lv = r["levels"] if isinstance(r["levels"], dict) \
                else json.loads(r["levels"])
            g = r["generations"] if isinstance(r["generations"], list) \
                else json.loads(r["generations"])
            key = tuple(int(np.sign(lv[k])) for k in tri)
            cells[(model, arm, r["trial_id"], tri)].setdefault(key, []).extend(g)
            stems[(model, arm, r["trial_id"])] = r.get("stem_id")
    load_generations.stems = stems
    return cells, paths


class Texts:
    """The three text conditions, derived once: raw, dedup, and coherent
    (raw text, degenerate generations dropped)."""

    def __init__(self):
        self._d, self._g = {}, {}

    def dedup(self, t):
        v = self._d.get(t)
        if v is None:
            v = self._d[t] = dedup(t)
        return v

    def degenerate(self, t):
        v = self._g.get(t)
        if v is None:
            v = self._g[t] = degenerate(t)
        return v

    def view(self, texts, cond):
        if cond == "dedup":
            return [self.dedup(t) for t in texts]
        if cond == "raw":
            return list(texts)
        if cond == "coherent":
            return [t for t in texts if not self.degenerate(t)]
        raise ValueError(cond)


def steering_gains(cells, emb, D, T, cond):
    """Per model and axis: align(raw arm) - align(control arm), main effect.

    PER-MODEL throughout: pooling every model's generations before
    differencing yields larger figures and is not what the paper reports.
    """
    gains, ctrl = {}, {}
    idx = defaultdict(lambda: defaultdict(lambda: ([], [])))
    for (m, arm, _, tri), cc in cells.items():
        if arm not in ("raw", "control"):
            continue
        for i, axis in enumerate(tri):
            hi, lo = idx[(m, axis)][arm]
            for key, g in cc.items():
                (hi if key[i] > 0 else lo).extend(T.view(g, cond))
    for (m, axis), arms in idx.items():
        if axis not in D or "raw" not in arms or "control" not in arms:
            continue
        res = {}
        for arm in ("raw", "control"):
            hi, lo = arms[arm]
            if len(hi) < 3 or len(lo) < 3:
                break
            res[arm] = float(unit(emb(hi).mean(0) - emb(lo).mean(0)) @ D[axis])
        if len(res) == 2:
            gains[(m, axis)] = res["raw"] - res["control"]
            ctrl[(m, axis)] = res["control"]
    return gains, ctrl


def axis_sets(cells, T, cond="dedup", arm="raw"):
    """High and low text sets per axis, pooled across models, in cell order -
    the input of the cross-alignment matrix and of its bootstrap."""
    hi, lo = defaultdict(list), defaultdict(list)
    for (m, a_, _, tri), cc in cells.items():
        if a_ != arm:
            continue
        for j, axis in enumerate(tri):
            for key, g in cc.items():
                (hi if key[j] > 0 else lo)[axis].extend(T.view(g, cond))
    return hi, lo


def cross_alignment(hi, lo, emb, D, ax):
    M = np.full((len(ax), len(ax)), np.nan)
    for i, axis in enumerate(ax):
        if len(hi[axis]) < 3 or len(lo[axis]) < 3:
            continue
        d = unit(emb(hi[axis]).mean(0) - emb(lo[axis]).mean(0))
        for j, other in enumerate(ax):
            M[i, j] = float(d @ D[other])
    return M


def per_model_summary(gains):
    models = sorted({m for m, _ in gains})
    return {m: float(np.mean([v for (mm, _), v in gains.items() if mm == m]))
            for m in models}


def check_steering(rep, cells, emb, D, T, ctx):
    """Section 6.2: gains over matched controls, per model, and the 7x7
    cross-alignment matrix. Primary text condition: dedup."""
    from scipy.stats import ttest_1samp, wilcoxon
    print("[steering] alignment against matched random controls")
    gains, ctrl = steering_gains(cells, emb, D, T, "dedup")
    ctx["gains_mpnet"] = gains
    S = "single_axis_steering"
    rep.check(S, "cells_total", len(gains), "model-axis cells")
    rep.check(S, "cells_positive", sum(v > 0 for v in gains.values()),
              "model-axis cells positive")
    pm = list(per_model_summary(gains).values())
    rep.check(S, "model_level_mean_gain", float(np.mean(pm)), "model-level mean gain")
    tt = ttest_1samp(pm, 0)
    rep.check(S, "model_level_t", float(tt.statistic), "  t(8)")
    rep.check(S, "model_level_p", float(tt.pvalue), "  p, two-sided")
    rep.check(S, "model_level_wilcoxon_p",
              float(wilcoxon(pm, alternative="greater").pvalue),
              "  Wilcoxon p, one-sided")
    by_axis = {a_: [v for (m, x), v in gains.items() if x == a_] for a_ in AXES}
    cby = {a_: [v for (m, x), v in ctrl.items() if x == a_] for a_ in AXES}
    for a_ in AXES:
        if by_axis[a_]:
            rep.check(S, ("gain_by_axis", a_), float(np.mean(by_axis[a_])),
                      f"  gain, {a_}")
    for a_ in AXES:
        if cby[a_]:
            rep.check(S, ("control_by_axis", a_), float(np.mean(cby[a_])),
                      f"  matched control, {a_}")
    rep.dump["control_by_axis"] = {a_: float(np.mean(cby[a_])) for a_ in AXES if cby[a_]}
    rep.check(S, "controls_negative",
              sum(1 for a_ in AXES if cby[a_] and np.mean(cby[a_]) < 0),
              "controls negative")
    rep.dump["gain_per_model"] = {}
    for (m, a_), v in gains.items():
        rep.dump["gain_per_model"].setdefault(m, {})[a_] = v

    # construct specificity: does each axis move its OWN construct most?
    hi, lo = axis_sets(cells, T)
    ax = [a_ for a_ in AXES if a_ in D]
    M = cross_alignment(hi, lo, emb, D, ax)
    ctx["crossalign"] = (M, ax, hi, lo)
    if np.isfinite(M).all():
        dm = sum(int(np.argmax(M[i]) == i) for i in range(len(ax)))
        for i, a_ in enumerate(ax):
            j = int(np.argmax(M[i]))
            if j != i:
                rep.note(f"    {a_:<20} own {M[i, i]:+.3f} but {ax[j]} "
                         f"is higher at {M[i, j]:+.3f}")
        off = (M.sum() - np.trace(M)) / (M.size - len(ax))
        rep.check(S, "cross_alignment_diagonal_max", dm,
                  "cross-alignment diagonal is max")
        rep.check(S, "cross_alignment_mean_diagonal", float(np.mean(np.diag(M))),
                  "  mean diagonal")
        rep.check(S, "cross_alignment_mean_offdiagonal", float(off),
                  "  mean off-diagonal")
        for (a_, b_) in (("valence", "intensity"), ("valence", "antagonism_peace"),
                         ("valence", "social_inner_outer"),
                         ("antagonism_peace", "valence"), ("intensity", "valence")):
            rep.check(S, ("cross_alignment_entry", f"{a_}->{b_}"),
                      float(M[ax.index(a_), ax.index(b_)]),
                      f"  steering {a_}, read on {b_}")
        rep.dump["cross_alignment"] = dict(axes=ax, M=M.tolist())


def check_robustness(rep, cells, emb, D, T, ctx):
    """Section 6.2's artefact table and section 9: gains under raw text,
    deduplicated text (primary) and coherent text (degenerate generations
    dropped). UNCAPPED, like verify's primary computation: embed_align's
    default cap of 400 per group gives slightly different gains."""
    print("[robustness] text conditions: raw, dedup, coherent")
    G = {c: steering_gains(cells, emb, D, T, c)[0]
         for c in ("raw", "dedup", "coherent")}
    ax_mean = {c: {a_: float(np.mean([v for (m, x), v in g.items() if x == a_]))
                   for a_ in AXES if any(x == a_ for (_, x) in g)}
               for c, g in G.items()}
    d_rd = max(abs(ax_mean["raw"][a_] - ax_mean["dedup"][a_]) for a_ in ax_mean["dedup"])
    d_dc = max(abs(ax_mean["dedup"][a_] - ax_mean["coherent"][a_])
               for a_ in ax_mean["dedup"])
    S = "robustness"
    rep.check(S, "raw_vs_dedup_max_delta", d_rd,
              "raw vs dedup, max |per-axis gain change|")
    rep.check(S, "dedup_vs_coherent_max_delta", d_dc,
              "dedup vs coherent, max |per-axis gain change|")
    rep.check(S, "coherent_axes_positive",
              sum(v > 0 for v in ax_mean["coherent"].values()),
              "axes positive, coherent text")
    cell = lambda a, b: max(abs(G[a][k] - G[b][k]) for k in G[b] if k in G[a])
    rep.note(f"    per model-axis cell instead of per axis: raw vs dedup "
             f"{cell('raw', 'dedup'):.3f}, dedup vs coherent "
             f"{cell('dedup', 'coherent'):.3f}")
    n_all = sum(len(g) for (m, a_, _, _), cc in cells.items()
                if a_ in ("raw", "control") for g in cc.values())
    n_deg = sum(T.degenerate(t) for (m, a_, _, _), cc in cells.items()
                if a_ in ("raw", "control") for g in cc.values() for t in g)
    rep.note(f"    degenerate generations, raw and control arms: "
             f"{n_deg}/{n_all} ({n_deg / max(n_all, 1):.1%})")
    rep.dump["robustness"] = ax_mean


def check_crossboot(rep, emb, D, ctx, n_boot=1200):
    """Port of bootstrap_crossalign.py: resample generations within each
    axis-level group (pooled across models, dedup text, raw arm) and recompute
    the cross-alignment matrix. The RNG is drawn in the script's order - per
    resample, every axis's high indices, then every axis's low indices - and
    the resampled means are formed as count-weighted sums, which is the same
    arithmetic without copying the matrices 1,200 times."""
    print(f"[crossboot] cross-alignment bootstrap, {n_boot} resamples")
    if "crossalign" not in ctx:
        rep.note("  needs the steering section; skipped")
        return
    M0, ax, hi, lo = ctx["crossalign"]
    H = {k: emb(hi[k]) for k in ax}
    L = {k: emb(lo[k]) for k in ax}
    Dm = np.array([D[k] for k in ax])
    rng = np.random.default_rng(SEED)
    wins = np.zeros(len(ax))
    chunk = 100
    for b0 in range(0, n_boot, chunk):
        nb = min(chunk, n_boot - b0)
        ch = {k: np.zeros((nb, len(H[k]))) for k in ax}
        cl = {k: np.zeros((nb, len(L[k]))) for k in ax}
        for b in range(nb):
            for k in ax:
                ch[k][b] = np.bincount(rng.integers(0, len(H[k]), len(H[k])),
                                       minlength=len(H[k]))
            for k in ax:
                cl[k][b] = np.bincount(rng.integers(0, len(L[k]), len(L[k])),
                                       minlength=len(L[k]))
        dirs = np.stack([(ch[k] @ H[k]) / len(H[k]) - (cl[k] @ L[k]) / len(L[k])
                         for k in ax], 1)                    # (nb, axes, dim)
        dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
        Mb = dirs @ Dm.T                                     # (nb, axes, axes)
        wins += (Mb.argmax(2) == np.arange(len(ax))[None, :]).sum(0)
    margins = []
    for i in range(len(ax)):
        others = [M0[i, j] for j in range(len(ax)) if j != i]
        margins.append(M0[i, i] - max(others))
    S = "single_axis_steering"
    rep.check(S, "bootstrap_min_row_wins", int(wins.min()),
              f"bootstrap: fewest resamples in which a row is dominant")
    rep.check(S, "bootstrap_min_margin", float(min(margins)),
              f"  smallest margin ({ax[int(np.argmin(margins))]})")
    rep.check(S, "bootstrap_max_margin", float(max(margins)),
              f"  largest margin ({ax[int(np.argmax(margins))]})")
    rep.dump["crossalign_boot"] = dict(axes=ax, wins=wins.tolist(),
                                       margins=[float(x) for x in margins])


def fit_trial(texts_by_key, emb, D, tri):
    """Per-trial transfer matrix as transfer_independent.py fits it: cell means
    with the trial's grand mean removed, projected on each probe, least squares
    with an intercept, TRANSPOSED - so rows are constructs, columns dials."""
    keys = sorted(texts_by_key)
    texts, spans = [], []
    for k in keys:
        spans.append((len(texts), len(texts) + len(texts_by_key[k])))
        texts.extend(texts_by_key[k])
    E = emb(texts)
    P = np.array([E[s:e].mean(0) for s, e in spans])
    P = P - P.mean(0, keepdims=True)
    Y = np.array([[p @ D[k] for k in tri] for p in P])
    L = np.array(keys, float)
    X = np.hstack([L, np.ones((len(L), 1))])
    return np.linalg.lstsq(X, Y, rcond=None)[0][:-1].T


def rownorm_stats(M):
    Mn = M / (np.abs(M).max(axis=1, keepdims=True) + 1e-12)
    iu, il = np.triu_indices(3, 1), np.tril_indices(3, -1)
    off = np.concatenate([np.abs(Mn[iu]), np.abs(Mn[il])]).mean()
    dm = [int(np.argmax(np.abs(Mn[i])) == i) for i in range(3)]
    return np.abs(np.diag(Mn)).mean() / max(off, 1e-9), dm


def check_control_and_dual(rep, cells, emb, D, T, slugs, ctx):
    """Sections 7.2-7.3, per repetition: each run of a triple is its own
    experiment, fitted on six generations per cell, dedup text."""
    from scipy.stats import wilcoxon
    print("[control] simultaneous control and the dual basis, per repetition")
    models = sorted({k[0] for k in cells})
    per_trial = defaultdict(list)
    for (m, arm, trial, tri), cc in cells.items():
        if arm in ("raw", "dual") and len(cc) >= 8 and all(k in D for k in tri):
            tb = {k: T.view(v, "dedup") for k, v in cc.items()}
            per_trial[(m, arm)].append((tri, fit_trial(tb, emb, D, tri)))
    ctx["per_trial"] = per_trial
    dm_all, ratio_all, by_axis = [], [], defaultdict(list)
    comp, rchg, n_sym, n_dia = [], [], 0, 0
    for m in models:
        Mr = per_trial.get((m, "raw"), [])
        if not Mr:
            continue
        rr, dm = [], []
        for tri, M in Mr:
            r_, d_ = rownorm_stats(M)
            rr.append(r_); dm += d_
            for i, a_ in enumerate(tri):
                by_axis[a_].append(d_[i])
        dm_all.append(float(np.mean(dm)))
        ratio_all.append(float(np.mean(rr)))
        slug = slug_for(m, slugs) or m
        rep.check("simultaneous_control", ("diag_is_max_by_model", slug),
                  dm_all[-1], f"  diag-is-max, {m[:24]}")
        Md = per_trial.get((m, "dual"), [])
        if Md:
            st = trio_stats([M for _, M in Mr], [M for _, M in Md])
            comp.append((st["diag"], st["sym"], st["anti"]))
            rchg.append(st["sep"])
            n_sym += st["sym"] < 0
            n_dia += st["diag"] < 0
    # Section 7.4: mean |off-diagonal| per axis pair over every repetition,
    # both directions (9 models x 5 triples x 2 repetitions x 2 = 180), on
    # row-normalised matrices. Which analysis produced the printed table is not
    # recorded; this is the one whose counts match the text.
    pair = defaultdict(list)
    for (m, arm), lst in per_trial.items():
        if arm != "raw":
            continue
        for tri, M in lst:
            Mn = M / (np.abs(M).max(axis=1, keepdims=True) + 1e-12)
            for i in range(3):
                for j in range(3):
                    if i != j:
                        pair["|".join(sorted((tri[i], tri[j])))].append(abs(Mn[i, j]))
    ptab = {k: float(np.mean(v)) for k, v in pair.items()}
    for k in sorted(ptab, key=ptab.get):
        if rep.spec("pair_crosstalk", ("pairs", k)):
            rep.check("pair_crosstalk", ("pairs", k), ptab[k], f"  pair {k} (n={len(pair[k])})")
    rep.dump["pair_crosstalk"] = {k: dict(mean=ptab[k], n=len(pair[k])) for k in ptab}
    rep.dump["per_trial_by_model"] = dict(
        models=[m for m in models if per_trial.get((m, "raw"))],
        diag_is_max=dm_all, ratio=ratio_all, components=comp, sep_change=rchg)
    S = "simultaneous_control"
    rep.check(S, "diag_is_max_mean", float(np.mean(dm_all)),
              "diagonal is row max (ROWS = CONSTRUCT), mean")
    rep.check(S, "diag_off_ratio_mean", float(np.mean(ratio_all)),
              "diag/off ratio, mean")
    rep.check(S, "models_above_chance", sum(x > 1 / 3 for x in dm_all),
              "models above chance")
    for a_ in AXES:
        if by_axis[a_]:
            rep.check(S, ("diag_is_max_by_axis", a_), float(np.mean(by_axis[a_])),
                      f"  per axis, {a_} (n={len(by_axis[a_])})")
    if comp:
        c = np.array(comp)
        S = "dual_basis"
        rep.note("  the negative result, verified with the same tolerances:")
        rep.check(S, "symmetric_offdiag_change", float(c[:, 1].mean()),
                  "  symmetric off-diag change")
        rep.check(S, "diagonal_change", float(c[:, 0].mean()), "  diagonal change")
        rep.check(S, "antisymmetric_offdiag_change", float(c[:, 2].mean()),
                  "  antisymmetric off-diag change")
        rep.check(S, "models_symmetric_reduced", int(n_sym), "  models, symmetric smaller")
        rep.check(S, "models_diagonal_reduced", int(n_dia), "  models, diagonal smaller")
        rep.check(S, "ratio_change", float(np.mean(rchg)),
                  "  per-trial separability x (noise-dominated)")
        rep.check(S, "models_ratio_improved", int(sum(x > 1 for x in rchg)),
                  "  models where per-trial ratio improved")
        rep.check(S, "wilcoxon_p_ratio_improved",
                  float(wilcoxon(np.array(rchg) - 1, alternative="greater").pvalue),
                  "  Wilcoxon p, improvement (one-sided)")
        ctx["observed_trio_sep"] = dict(zip([m for m in models
                                             if per_trial.get((m, "dual"))], rchg))


def check_magnitude(rep, cells, emb, D, probes, T, ctx):
    """Port of transfer_matrix.py, which the paper quotes for the POOLED
    diagonal dominance (91.5%) and for section 6.3's magnitudes.

    Its choices, reproduced as written, differ from verify's own in three
    ways, and all three matter for reading the numbers:
      - both repetitions of a triple are pooled into one fit (12 generations
        per cell rather than 6);
      - text is RAW, not deduplicated;
      - positions are unit-normalised cell means before projection.
    The same pooling with verify's own conventions is printed beside it, so the
    effect of pooling alone can be read off.

    'corpus_fraction' divides by the range of THIS TRIPLE'S GENERATIONS along
    the probe, not by the range of the emotion corpus as section 6.3's formula
    states. Ported as computed; see the report.
    """
    from scipy.stats import spearmanr
    print("[magnitude] pooled transfer and magnitude (port of transfer_matrix.py)")
    pooled = defaultdict(lambda: defaultdict(list))   # (m, tri) -> key -> texts
    ntr = defaultdict(set)
    for (m, arm, trial, tri), cc in cells.items():
        if arm != "raw" or any(k not in D for k in tri):
            continue
        ntr[(m, tri)].add(trial)
        for key, g in cc.items():
            pooled[(m, tri)][key].extend(g)
    pole_e = {k: np.linalg.norm(emb(probes[k]["pos"]).mean(0)
                                - emb(probes[k]["neg"]).mean(0)) for k in AXES}
    per_model = defaultdict(lambda: dict(ok=0, rows=0, own_ok=0,
                                         mag=defaultdict(lambda: defaultdict(list))))
    triple_pf = []
    for (m, tri), cc in sorted(pooled.items()):
        if len(cc) < 8 or len(ntr[(m, tri)]) < 2:     # transfer_matrix --min-trials 2
            continue
        keys = sorted(cc)
        texts, spans = [], []
        for k in keys:
            spans.append((len(texts), len(texts) + len(cc[k])))
            texts.extend(cc[k])
        E = emb(texts)
        Pm = np.array([E[s:e].mean(0) for s, e in spans])
        Pm = Pm - Pm.mean(0, keepdims=True)
        Y = np.array([[unit(p) @ D[k] for k in tri] for p in Pm])
        for ci, k in enumerate(tri):
            hi = np.array([Pm[i] for i, kk in enumerate(keys) if kk[ci] > 0])
            lo = np.array([Pm[i] for i, kk in enumerate(keys) if kk[ci] < 0])
            disp = float((hi.mean(0) - lo.mean(0)) @ D[k])
            proj = (E - E.mean(0)) @ D[k]
            sd, rng_ = float(np.std(proj)), float(np.ptp(proj))
            pf = disp / pole_e[k] if pole_e[k] > 0 else np.nan
            per_model[m]["mag"][k]["pole"].append(pf)
            per_model[m]["mag"][k]["corpus"].append(disp / rng_ if rng_ > 0 else np.nan)
            per_model[m]["mag"][k]["d"].append(disp / sd if sd > 0 else np.nan)
            triple_pf.append(pf)
        L = np.array(keys, float)
        Lc = L - L.mean(0, keepdims=True)
        M = np.linalg.lstsq(Lc, Y, rcond=None)[0].T          # rows: construct
        Mn = M / (np.abs(M).max(axis=1, keepdims=True) + 1e-12)
        per_model[m]["ok"] += sum(int(np.argmax(np.abs(Mn[i])) == i) for i in range(3))
        per_model[m]["rows"] += 3
        # the same pooling in verify's own conventions (dedup, no unit, intercept)
        Mv = fit_trial({k: T.view(v, "dedup") for k, v in cc.items()}, emb, D, tri)
        per_model[m]["own_ok"] += sum(rownorm_stats(Mv)[1])
    if not per_model:
        rep.note("  nothing to pool; skipped")
        return
    S = "simultaneous_control"
    rates = [v["ok"] / v["rows"] for v in per_model.values()]
    rep.dump["pooled_diag_by_model"] = {m: v["ok"] / v["rows"]
                                        for m, v in per_model.items()}
    rep.dump["magnitude_by_model"] = {
        m: {k: {q: float(np.nanmean(x)) for q, x in mm.items()}
            for k, mm in v["mag"].items()} for m, v in per_model.items()}
    own = [v["own_ok"] / v["rows"] for v in per_model.values()]
    rep.check(S, "diag_is_max_pooled", float(np.mean(rates)),
              "diag is row max, repetitions POOLED (transfer_matrix port)")
    rep.note(f"    the same pooling with verify's conventions (dedup text, "
             f"no unit-normalisation): {np.mean(own):.3f}")
    agg = defaultdict(lambda: defaultdict(list))
    for v in per_model.values():
        for k, mm in v["mag"].items():
            for q in ("pole", "corpus", "d"):
                agg[k][q].append(float(np.nanmean(mm[q])))
    S = "magnitude"
    tab = {}
    for k in AXES:
        if k not in agg:
            continue
        tab[k] = {q: float(np.mean(agg[k][q])) for q in ("pole", "corpus", "d")}
        rep.check(S, ("pole_fraction", k), tab[k]["pole"], f"  pole fraction, {k}")
        rep.check(S, ("corpus_fraction", k), tab[k]["corpus"], f"  corpus fraction, {k}")
        rep.check(S, ("cohens_d", k), tab[k]["d"], f"  Cohen's d, {k}")
    ks = list(tab)
    rep.check(S, "pole_corpus_rho",
              float(spearmanr([tab[k]["pole"] for k in ks],
                              [tab[k]["corpus"] for k in ks]).statistic),
              "pole vs corpus fraction, Spearman rho")
    ds = [tab[k]["d"] for k in ks]
    rep.check(S, "cohens_d_min", float(min(ds)), "Cohen's d, min")
    rep.check(S, "cohens_d_max", float(max(ds)), "Cohen's d, max")
    rep.check(S, "axes_d_above_1", int(sum(x > 1.0 for x in ds)), "axes with d > 1.0")
    tp = np.array(triple_pf)
    rep.note(f"    per triple and axis (what fig6_spread.png plots): "
             f"{(tp < 0.05).mean():.1%} below 5% of pole distance, n={len(tp)}")
    rep.dump["magnitude"] = tab


def check_spread(rep, cells, emb, D, probes, T, slugs):
    """Appendix A and section 9: displacement per MATCHED PAIR - two cells of
    one trial that differ only in this axis's level - as a fraction of the
    probe's pole distance. Dedup text.

    RECONSTRUCTION: the script that wrote steering_examples_cstar.md is not in
    this package, so the pairing and text condition here are inferred from the
    counts the paper gives (1,080 pairs per axis = 9 models x 30 trials x 4
    cell pairs). Treat a failure here as a question about the reconstruction
    before treating it as one about the paper.
    """
    print("[spread] matched-pair displacement (reconstruction)")
    pole_e = {k: np.linalg.norm(emb(probes[k]["pos"]).mean(0)
                                - emb(probes[k]["neg"]).mean(0)) for k in AXES}
    pf = defaultdict(list)
    best = {}
    for (m, arm, _, tri), cc in cells.items():
        if arm != "raw" or len(cc) < 8:
            continue
        mean = {k: emb(T.view(v, "dedup")).mean(0) for k, v in cc.items()}
        for ci, k in enumerate(tri):
            if k not in D:
                continue
            for key in cc:
                if key[ci] < 0:
                    continue
                partner = tuple(-x if i == ci else x for i, x in enumerate(key))
                v = float((mean[key] - mean[partner]) @ D[k]) / pole_e[k]
                pf[k].append(v)
                if v > best.get((m, k), -np.inf):
                    best[(m, k)] = v
    S = "spread"
    for k in AXES:
        if not pf[k]:
            continue
        rep.check(S, ("pairs", k), len(pf[k]), f"  matched pairs, {k}")
    med = {k: float(np.median(v)) for k, v in pf.items()}
    lo_k, hi_k = min(med, key=med.get), max(med, key=med.get)
    rep.check(S, "median_min", med[lo_k], f"median, lowest axis ({lo_k})")
    rep.check(S, "median_max", med[hi_k], f"median, highest axis ({hi_k})")
    rep.check(S, "medians_positive", sum(v > 0 for v in med.values()),
              "axes with positive median")
    for k in ("heat", "arousal"):
        if pf[k]:
            rep.check(S, ("min", k), float(min(pf[k])), f"  most negative pair, {k}")
    rep.check(S, "axes_with_negative_pairs", sum(min(v) < 0 for v in pf.values()),
              "axes with pairs moving the wrong way")
    allv = np.concatenate([np.array(v) for v in pf.values()])
    rep.check(S, "share_below_5pct", float((allv < 0.05).mean()),
              "share of pairs below 5% of pole distance")
    g2 = next((m for m in {m for m, _ in best} if "gemma-2-9b" in m), None)
    if g2:
        rep.check(S, "gemma2_antagonism_max", best[(g2, "antagonism_peace")],
                  "strongest antagonism pair, Gemma-2-9B")
    rep.dump["spread_medians"] = med


def check_superposition(rep, cells, scells, emb, D, grams, slugs, T, ctx):
    """Port of superposition_test.py plus the pooled statistics of
    make_figures.fig7. ROWS = DIAL throughout.

    Per trial: M_obs from the eight three-dial cells, R from the six single-dial
    cells of the same trial and stem, M_pred from R under linearity with the
    harness's per-cell rescaling. Pooled: each ordered pair's entry averaged
    over every trial containing it, into a 7x7 per model and arm.
    """
    from scipy.stats import wilcoxon
    print("[superposition] single dials predict three dials?")
    L8 = np.array([[(1 if (c >> b) & 1 else -1) for b in range(3)]
                   for c in range(8)], float)
    Tr = defaultdict(dict)
    Si = defaultdict(dict)
    for (m, arm, tid, tri), cc in cells.items():
        if arm in ("raw", "dual"):
            Tr[(m, arm)][tid] = (tri, cc)
    for (m, arm, tid, tri), cc in scells.items():
        if arm in ("raw", "dual"):
            Si[(m, arm)][tid] = (tri, cc)
    models = sorted({m for m, a_ in Tr if (m, a_) in Si})
    # Guards. The prediction is only a test of superposition if the single-dial
    # cells were generated on the same stems, at the same magnitude, as the
    # three-dial cells they predict. A single-dial run from another plan or at
    # another c produces plausible numbers and a failed superposition - which
    # is how a wrong folder in the package first showed itself.
    st3, st1 = ctx.get("stems3", {}), ctx.get("stems1", {})
    bad_stem = sum(1 for (m, arm, t), sid in st1.items()
                   if (m, arm, t) in st3 and st3[(m, arm, t)] != sid)
    rep.check("superposition", "trials_stem_mismatch", bad_stem,
              "single-dial trials on a different stem from their three-dial trial")
    cst = rep.exp.get("coherence_edges", {}).get("c_star", {})
    for mp in sorted(ctx.get("single_manifests", [])):
        mf = json.load(open(mp, encoding="utf-8"))
        slug = str(mf.get("model", "")).replace("/", "_")
        want_c = cst.get(slug)
        ok = (mf.get("design") == "single" and want_c is not None
              and abs(float(mf.get("c", -1)) - want_c) < 1e-9)
        rep.rows.append((f"  single-dial run at c*, {slug[:30]}", want_c, mf.get("c"),
                         0, "PASS" if ok else "FAIL"))
    if not ctx.get("single_manifests"):
        rep.note("  no single-dial manifests shipped; magnitude of the "
                 "single-dial runs not checked (rerun package_verify --add-only)")
    res = {}
    for m in models:
        slug = slug_for(m, slugs)
        if slug not in grams:
            rep.note(f"  {m}: no Gram (vectors missing); skipped")
            continue
        G = grams[slug]; Gi = np.linalg.inv(G)
        per = {}
        for arm in ("raw", "dual"):
            rows = []
            for tid, (tri, cc) in Tr.get((m, arm), {}).items():
                if tid not in Si[(m, arm)] or len(cc) < 8:
                    continue
                stri, sc = Si[(m, arm)][tid]
                if tuple(stri) != tuple(tri) or len(sc) < 6:
                    continue
                idx = [AXES.index(k) for k in tri]
                D3 = np.array([D[k] for k in tri])
                read = lambda texts: D3 @ emb(T.view(texts, "dedup")).mean(0)
                Y = np.array([read(cc[tuple(int(x) for x in l)]) for l in L8])
                M_obs = (L8.T @ Y) / 8.0
                R = np.array([(read(sc[tuple(1 if j == i else 0 for j in range(3))])
                               - read(sc[tuple(-1 if j == i else 0 for j in range(3))])) / 2
                              for i in range(3)])
                K = G[np.ix_(idx, idx)] if arm == "raw" else Gi[np.ix_(idx, idx)]
                w = np.ones(3) if arm == "raw" else np.sqrt(np.diag(Gi)[idx])
                Yp = np.array([(l * w / np.sqrt(l @ K @ l)) @ R for l in L8])
                rows.append(dict(triple=list(tri), M_obs=M_obs,
                                 M_pred=(L8.T @ Yp) / 8.0, single=R))
            per[arm] = rows
        if not per.get("raw") or not per.get("dual"):
            continue

        def corr(a, b):
            return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])
        byt = defaultdict(list)
        for r in per["raw"]:
            byt[tuple(r["triple"])].append(r["M_obs"])
        pairs = [v[:2] for v in byt.values() if len(v) >= 2]

        def pooled(rows, key):
            acc = defaultdict(list)
            for t in rows:
                for i, x in enumerate(t["triple"]):
                    for j, y in enumerate(t["triple"]):
                        acc[(AXES.index(x), AXES.index(y))].append(t[key][i, j])
            return np.array([[np.mean(acc[(i, j)]) for j in range(7)]
                             for i in range(7)])
        g = {key: sep7(pooled(per["dual"], key)) / sep7(pooled(per["raw"], key))
             for key in ("single", "M_obs", "M_pred")}
        res[m] = dict(
            agreement=corr([r["M_pred"] for r in per["raw"]],
                           [r["M_obs"] for r in per["raw"]]),
            ceiling=corr([p[0] for p in pairs], [p[1] for p in pairs])
            if pairs else float("nan"),
            gain_single=g["single"], gain_obs=g["M_obs"], gain_pred=g["M_pred"],
            anti_single=anti7(pooled(per["raw"], "single")),
            anti_three=anti7(pooled(per["raw"], "M_obs")),
            n_trials=len(per["raw"]))
    if not res:
        rep.note("  no model with both designs; skipped")
        return
    V = lambda k: np.array([r[k] for r in res.values()])
    S = "superposition"
    rep.check(S, "agreement_mean", float(V("agreement").mean()),
              "prediction vs observed, r (mean over models)")
    rep.check(S, "ceiling_mean", float(np.nanmean(V("ceiling"))),
              "  repetition ceiling, r")
    for key, lab in (("gain_single", "one dial at a time"),
                     ("gain_obs", "three at once, observed"),
                     ("gain_pred", "three at once, predicted")):
        rep.check(S, f"{key}_mean", float(V(key).mean()),
                  f"pooled dual gain x, {lab}")
    for key, lab in (("gain_single", "single"), ("gain_obs", "three observed")):
        rep.check(S, f"{key}_models_improved", int((V(key) > 1).sum()),
                  f"  models improved, {lab}")
        rep.check(S, f"{key}_wilcoxon_p",
                  float(wilcoxon(V(key) - 1, alternative="greater").pvalue),
                  f"  Wilcoxon p (one-sided), {lab}")
    rep.check(S, "pred_obs_r_across_models",
              float(np.corrcoef(V("gain_pred"), V("gain_obs"))[0, 1]),
              "predicted vs observed three-dial gain, r across models")
    rep.check(S, "anti_single_mean", float(V("anti_single").mean()),
              "antisym share, single dial, pooled")
    rep.check(S, "anti_single_min", float(V("anti_single").min()), "  min over models")
    rep.check(S, "anti_single_max", float(V("anti_single").max()), "  max over models")
    rep.check(S, "anti_three_mean", float(V("anti_three").mean()),
              "antisym share, three at once, pooled")
    prim = next((m for m in res if slug_for(m, slugs) == PRIMARY_MODEL), None)
    if prim:
        rep.check(S, "primary_gain_single", res[prim]["gain_single"],
                  "Gemma-3-1B-it: observed single-dial gain x")
    gp = ctx.get("geompred", {})
    if gp:
        rep.note("    single dial, per model:   predicted (geometry)  observed"
                 "   antisym geometric  observed")
        pr, ob = [], []
        for m, r in res.items():
            s_ = slug_for(m, slugs)
            if s_ in gp:
                pr.append(gp[s_]["pred_gain"]); ob.append(r["gain_single"])
                rep.note(f"    {m[:26]:<26}{gp[s_]['pred_gain']:>12.2f}"
                         f"{r['gain_single']:>12.2f}"
                         f"{gp[s_]['anti_share_geometric']:>18.0%}"
                         f"{r['anti_single']:>11.0%}")
                rep.dump.setdefault("single_dial_table", {})[m] = dict(
                    predicted=gp[s_]["pred_gain"], observed=r["gain_single"],
                    anti_geometric=gp[s_]["anti_share_geometric"],
                    anti_geometric_three=gp[s_]["anti_share_geometric_three"],
                    anti_observed=r["anti_single"], anti_observed_three=r["anti_three"])
        if len(pr) > 2:
            rep.note(f"    mean predicted x{np.mean(pr):.2f} against observed "
                     f"x{np.mean(ob):.2f}; observed below predicted in "
                     f"{sum(o < p for o, p in zip(ob, pr))}/{len(pr)}")
    rep.dump["superposition"] = res


def check_encoder2(rep, cells, emb2, probes, T, ctx):
    """Section 6.2's third artefact row: the steering measurement repeated with
    bge-large-en-v1.5. --full only; it re-embeds every raw and control
    generation with a large encoder."""
    from scipy.stats import ttest_1samp
    print(f"[encoder2] second encoder, {emb2.name}")
    D2 = {k: emb2.contrast(s["pos"], s["neg"]) for k, s in probes.items()
          if len(s.get("pos", [])) >= 2 and len(s.get("neg", [])) >= 2}
    gains, _ = steering_gains(cells, emb2, D2, T, "dedup")
    pm = list(per_model_summary(gains).values())
    S = "second_encoder"
    rep.check(S, "model_level_mean_gain", float(np.mean(pm)), "bge: model-level mean gain")
    rep.check(S, "model_level_t", float(ttest_1samp(pm, 0).statistic), "  bge: t(8)")
    rep.check(S, "cells_positive", sum(v > 0 for v in gains.values()),
              "  bge: cells positive")
    g1 = ctx.get("gains_mpnet")
    if g1:
        rep.check(S, "sign_agreement", sum((g1[k] > 0) == (gains[k] > 0)
                                           for k in gains if k in g1),
                  "  same sign as mpnet, cells")
        a1 = [np.mean([v for (m, x), v in g1.items() if x == a_]) for a_ in AXES]
        a2 = [np.mean([v for (m, x), v in gains.items() if x == a_]) for a_ in AXES]
        rho, p = exact_spearman_p(a1, a2)
        rep.check(S, "ordering_rho", rho, "  axis ordering, Spearman rho")
        rep.check(S, "ordering_p_exact", p, "  exact p (two-sided, 5,040 rankings)")
        rk1 = np.argsort(np.argsort(-np.array(a1)))
        rk2 = np.argsort(np.argsort(-np.array(a2)))
        rep.note("    per-axis mean gain, mpnet / bge (rank): " + ", ".join(
            f"{a_} {x:.2f}/{y:.2f} ({r1 + 1}/{r2 + 1})"
            for a_, x, y, r1, r2 in zip(AXES, a1, a2, rk1, rk2)))
    hi, lo = axis_sets(cells, T)
    ax = [a_ for a_ in AXES if a_ in D2]
    M = cross_alignment(hi, lo, emb2, D2, ax)
    if np.isfinite(M).all():
        rep.check(S, "cross_alignment_diagonal_max",
                  sum(int(np.argmax(M[i]) == i) for i in range(len(ax))),
                  "  bge: cross-alignment diagonal is max")


# ─────────────────────────────────────────────── main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=SECTIONS + FULL_ONLY)
    ap.add_argument("--full", action="store_true",
                    help="also the second encoder (bge-large-en-v1.5); slow on CPU")
    ap.add_argument("--encoder", default=MPNET)
    ap.add_argument("--encoder2", default=BGE)
    ap.add_argument("--n-boot", type=int, default=1200)
    ap.add_argument("--cache", default=None,
                    help="folder for an embedding cache; reruns skip encoding")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dump", default=None,
                    help="write the recomputed per-model values to this JSON")
    a = ap.parse_args()

    if a.list:
        print("  ".join(SECTIONS) + "   |  --full adds: " + "  ".join(FULL_ONLY))
        return

    exp = json.load(open(os.path.join(HERE, "expected.json"), encoding="utf-8"))
    rep = Report(exp)
    probes = json.load(open(os.path.join(DATA, "probes.json"), encoding="utf-8"))
    labels = load_labels(os.path.join(DATA, "labels.csv"))
    want = set(a.only) if a.only else set(SECTIONS) | (set(FULL_ONLY) if a.full else set())
    if a.full and a.only:
        want |= set(FULL_ONLY)
    t0 = time.time()
    print(f"verify.py — {exp['_paper']}")
    print(f"  {exp['_protocol']}")

    lsel = {}
    lp = os.path.join(DATA, "layer_selection.csv")
    if os.path.exists(lp):
        for r in csv.DictReader(open(lp, newline="", encoding="utf-8")):
            if r.get("selected") == "1":
                lsel[r["model"]] = int(r["layer"])
    slugs = sorted(set(lsel) | set(exp.get("coherence_edges", {}).get("c_star", {})))
    ctx = {}
    geo = None
    if "geometry" in want or "interface" in want:
        geo = check_geometry(rep, probes, labels, lsel) if "geometry" in want else None
    if "nulls" in want:
        check_nulls(rep, probes, labels, lsel)
    if "validation" in want:
        check_validation(rep, probes, labels, lsel)
    if "commonmag" in want:
        check_commonmag(rep)
    if "edges" in want:
        check_edges(rep)
    if "sharedwords" in want:
        check_shared_words(rep, probes, labels, lsel)
    if "interface" in want:
        check_interface(rep, geo)

    if want & NEEDS_ENCODER:
        emb = Embedder(a.encoder, a.cache)
        words = sorted({w for s in probes.values() for side in ("pos", "neg")
                        for w in s.get(side, [])})
        emb.prime(words)
        D = {k: emb.contrast(s["pos"], s["neg"]) for k, s in probes.items()
             if len(s.get("pos", [])) >= 2 and len(s.get("neg", [])) >= 2}
        grams = {}
        if want & {"linear", "geompred", "superposition"}:
            for m in slugs:
                g = model_gram(m, probes, labels, lsel)
                if g is not None:
                    grams[m] = g
            print(f"  activation Grams for {len(grams)} models")
        A = None
        if want & {"linear", "geompred"}:
            A = check_linear(rep, probes, labels, lsel, emb, grams) \
                if "linear" in want else np.array(
                    [[D[i] @ D[j] for j in AXES] for i in AXES])
        if "geompred" in want:
            check_geompred(rep, A, grams, ctx)

        if want & NEEDS_GENERATIONS:
            cells, paths = load_generations(os.path.join(DATA, "generations"))
            if not paths:
                rep.note("no generations found; generation checks skipped")
            else:
                T = Texts()
                print(f"  {len(paths)} generation files, "
                      f"{len({k[0] for k in cells})} models")
                scells = {}
                if "superposition" in want:
                    ctx["stems3"] = dict(load_generations.stems)
                    scells, sp = load_generations(
                        os.path.join(DATA, "generations_single"))
                    ctx["stems1"] = dict(load_generations.stems)
                    import glob as _gl
                    ctx["single_manifests"] = _gl.glob(os.path.join(
                        DATA, "generations_single", "*_manifest.json"))
                    bad = [k for k, cc in scells.items()
                           if any(sum(x != 0 for x in key) != 1 for key in cc)]
                    if bad:
                        sys.exit(f"generations_single/ holds {len(bad)} trials "
                                 f"that are not single-dial (e.g. {bad[0][:3]}); "
                                 f"the three-dial files were probably copied there.")
                    print(f"  {len(sp)} single-dial files")
                # every text any selected section will embed, in one pass
                need = []
                for (m, arm, _, _), cc in cells.items():
                    for g in cc.values():
                        need += T.view(g, "dedup")
                        if arm in ("raw", "control") and \
                                want & {"robustness", "magnitude"}:
                            need += g
                for cc in scells.values():
                    for g in cc.values():
                        need += T.view(g, "dedup")
                emb.prime(need)
                del need
                if want & {"steering", "crossboot", "encoder2"}:
                    check_steering(rep, cells, emb, D, T, ctx)
                if "robustness" in want:
                    check_robustness(rep, cells, emb, D, T, ctx)
                if "crossboot" in want:
                    check_crossboot(rep, emb, D, ctx, a.n_boot)
                if "control" in want:
                    check_control_and_dual(rep, cells, emb, D, T, slugs, ctx)
                if "magnitude" in want:
                    check_magnitude(rep, cells, emb, D, probes, T, ctx)
                if "spread" in want:
                    check_spread(rep, cells, emb, D, probes, T, slugs)
                if "superposition" in want:
                    if not scells:
                        rep.note("  data/generations_single/ absent; "
                                 "superposition skipped")
                    else:
                        check_superposition(rep, cells, scells, emb, D, grams,
                                            slugs, T, ctx)
                if "encoder2" in want:
                    emb2 = Embedder(a.encoder2, a.cache)
                    need = [t for (m, arm, _, _), cc in cells.items()
                            if arm in ("raw", "control")
                            for g in cc.values() for t in T.view(g, "dedup")]
                    emb2.prime(sorted({w for s in probes.values()
                                       for side in ("pos", "neg")
                                       for w in s.get(side, [])}) + need)
                    del need
                    check_encoder2(rep, cells, emb2, probes, T, ctx)

    fails = rep.table()
    if a.dump:
        def clean(o):
            if isinstance(o, dict):
                return {str(k) if not isinstance(k, tuple) else "|".join(map(str, k)):
                        clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return o
        json.dump(clean(rep.dump), open(a.dump, "w"), indent=1)
        print(f"\n  recomputed values -> {a.dump}")
    print(f"\n  {time.time() - t0:.0f} s")
    if fails:
        print("\n  A failure is not necessarily an error in the paper. Check the")
        print("  recomputed value against the tolerance in expected.json: a small")
        print("  excess is usually a library difference, a large one is not.")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
