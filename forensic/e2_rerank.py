"""E2 rerank: word-level candidate lattice + large native-script language model + trigram Viterbi.

The word-level system (e2_normaliser_plus + e2_neural) is capped by how often it *picks* the right reading, not by
whether the right reading is in its candidate set. So: (1) widen the lattice (seen readings + char-transformer beam-5
+ copy) and measure oracle recall; (2) replace the 48k-sentence in-domain bigram with a Kneser-Ney bigram trained on
IndicCorp v2 (3.4M lines) + Assamese Wikipedia + one Sangraha shard (~60M tokens), interpolated with the in-domain
model; (3) decode with a second-order (trigram) Viterbi whose trigram component is in-domain Kneser-Ney. Weights are
tuned on 2,000 held-out training sentences; the 12,063-sentence test split is identical to every other E2 script.
"""
import os, sys, json, math, pickle, time, glob, re
from collections import Counter, defaultdict
import numpy as np
os.environ.setdefault("D", "256"); os.environ.setdefault("LAYERS", "2"); os.environ.setdefault("BEAM", "5")
import torch
import e2_neural as N
import e2_normaliser_plus as V
from e2_romanisation import load, PATH

OUT = sys.argv[2] if len(sys.argv) > 2 else "results"; CACHE = "../data/new/lm_cache"; os.makedirs(CACHE, exist_ok=True)
SEED, DISC, K_SEEN = 0, 0.75, 8
NATIVE = re.compile(r"[ঀ-৿]")


class KN:
    """Interpolated Kneser-Ney n-gram LM over integer word ids (orders 1–3), built from a stream of token-id lists."""

    def __init__(self, order=2):
        self.order = order; self.c = [Counter() for _ in range(order + 1)]; self.hist = [Counter() for _ in range(order + 1)]; self.foll = [Counter() for _ in range(order + 1)]
        self.cont = Counter(); self.cont_hist = Counter(); self.V = 0

    def add(self, ids):
        ids = [-1] * (self.order - 1) + ids + [-2]
        for i in range(self.order - 1, len(ids)):
            for n in range(1, self.order + 1):
                g = tuple(ids[i - n + 1:i + 1]); h = g[:-1]
                if self.c[n][g] == 0 and n >= 2:
                    self.foll[n][h] += 1                                   # distinct followers of h
                    if n == 2: self.cont[g[-1]] += 1                       # distinct left contexts of w (continuation count)
                self.c[n][g] += 1; self.hist[n][h] += 1
        return self

    def finish(self):
        self.V = len(self.c[1]) + 1; self.cont_total = sum(self.cont.values()); return self

    def prob(self, h, w):
        """P(w | h) with h a tuple of up to order-1 ids; recursive interpolated KN; unknown words get continuation mass."""
        p = (self.cont.get(w, 0) + 0.5) / (self.cont_total + 0.5 * self.V)  # unigram continuation prob, add-0.5 for OOV
        for n in range(2, self.order + 1):
            hh = h[-(n - 1):] if n - 1 <= len(h) else None
            if hh is None or len(hh) < n - 1: break
            ch = self.hist[n].get(hh, 0)
            if ch == 0: continue
            p = max(self.c[n].get(hh + (w,), 0) - DISC, 0) / ch + DISC * self.foll[n][hh] / ch * p
        return p


def build_big_lm(vocab):
    """Bigram KN on the large corpora; vocab maps word -> id and is extended as new words appear."""
    f = f"{CACHE}/big_bigram.pkl"
    if os.path.exists(f): return pickle.load(open(f, "rb"))
    lm = KN(2); t0 = time.time(); n = 0
    def feed(line):
        nonlocal n
        toks = [t for t in line.split() if NATIVE.search(t)]
        if len(toks) >= 2: lm.add([vocab.setdefault(t, len(vocab)) for t in toks]); n += len(toks)
    for line in open("../data/new/indiccorp/data/as.txt", encoding="utf-8", errors="ignore"): feed(line)
    print(f"  IndicCorp: {n / 1e6:.1f}M tokens ({time.time() - t0:.0f}s)", flush=True)
    for line in open("../data/text/wiki100k_sentences.txt", encoding="utf-8", errors="ignore"): feed(line.split("\t", 1)[-1])
    import pyarrow.parquet as pq
    for doc in pq.read_table(sorted(glob.glob("../data/new/sangraha/verified/asm/*.parquet"))[0], columns=["text"]).column("text").to_pylist():
        for line in doc.split("\n"): feed(line)
    print(f"  total {n / 1e6:.1f}M tokens, vocab {len(vocab)}, bigrams {len(lm.c[2]) / 1e6:.1f}M ({time.time() - t0:.0f}s)", flush=True)
    lm.finish(); pickle.dump((lm, vocab), open(f, "wb")); return lm, vocab


def main():
    os.makedirs(OUT, exist_ok=True)
    pairs, _ = load(PATH); rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(pairs)); cut = int(0.8 * len(pairs)); tr, te = [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]
    dev, tr_lm = tr[:2000], tr[2000:]  # the word-level decoder uses all of tr (as every E2 script did); LMs/weights tuned on dev use tr_lm
    vocab = {}
    print("building large LM", flush=True); big, vocab = build_big_lm(vocab)
    ind2, ind3 = KN(2), KN(3)
    for _, n in tr_lm:
        ids = [vocab.setdefault(t, len(vocab)) for t in n]; ind2.add(ids); ind3.add(ids)
    ind2.finish(); ind3.finish(); print(f"in-domain LMs: {sum(ind2.c[1].values())} tokens", flush=True)

    # word-level candidate generators: seen readings (direct model) and the char transformer with beam 5
    dec = V.Decoder().fit(tr); ck = torch.load(f"{OUT}/e2_translit_big.pt", map_location="cpu")
    sv, tv = N.Vocab([]), N.Vocab([]); sv.itos, tv.itos = ck["sv"], ck["tv"]; sv.stoi, tv.stoi = {c: i for i, c in enumerate(sv.itos)}, {c: i for i, c in enumerate(tv.itos)}
    model = N.Translit(len(sv.itos), len(tv.itos)); model.load_state_dict(ck["model"]); model.eval()
    toks = sorted({a.lower() for r, _ in dev + te for a in r}); t0 = time.time(); N.predict(model, sv, tv, toks); neural = dict(zip(toks, N.predict.alts)); print(f"neural beam-{N.BEAM} for {len(toks)} tokens ({time.time() - t0:.0f}s)", flush=True)

    def cands(tok):
        t = tok.lower(); out = {}
        if t in dec.reads:
            c = dec.reads[t]; tot = sum(c.values())
            for b, k in c.most_common(K_SEEN): out[b] = math.log(k / tot)
            out.setdefault(tok, math.log(0.5 / (tot + 1)))
        hyps = neural[t]; top = hyps[0][1]
        for p, lp in hyps:
            sc = (lp - top) * 0.5 + top * 0.3 + math.log(0.6 if t not in dec.reads else 0.15); out[p] = max(out.get(p, -1e9), sc)
        if t not in dec.reads: out[tok] = max(out.get(tok, -1e9), math.log(0.3) - 0.5)
        return out

    def oracle(data):
        hit = tot = 0
        for r, n in data:
            for a, b in zip(r, n): tot += 1; hit += b in cands(a)
        return hit / tot

    def lm_logp(h, w, mu, nu):
        """Interpolated: in-domain trigram (weight nu) over a bigram mixture of in-domain (mu) and big (1-mu)."""
        wid = vocab.get(w, -3); hid = tuple(vocab.get(x, -3) for x in h)
        p2 = mu * ind2.prob(hid[-1:], wid) + (1 - mu) * big.prob(hid[-1:], wid)
        if nu > 0 and len(hid) == 2: p2 = nu * ind3.prob(hid, wid) + (1 - nu) * p2
        return math.log(p2)

    def decode(toks, lam, mu, nu, order):
        C = [cands(t) for t in toks]
        if order == 2:
            score, back = {V.BOS: 0.0}, []
            for cd in C:
                new, bp = {}, {}
                for w, em in cd.items():
                    best = max(score, key=lambda p: score[p] + lam * lm_logp((p,), w, mu, 0)); new[w] = score[best] + lam * lm_logp((best,), w, mu, 0) + em; bp[w] = best
                score, back = new, back + [bp]
            w = max(score, key=score.get); out = [w]
            for bp in reversed(back[1:]): w = bp[w]; out.append(w)
            return out[::-1]
        # second-order Viterbi: state = (prev, cur)
        states = {(V.BOS, w): em + lam * lm_logp((V.BOS,), w, mu, 0) for w, em in C[0].items()}; back = []
        for cd in C[1:]:
            new, bp = {}, {}
            for w, em in cd.items():
                for (u, v), s in states.items():
                    sc = s + em + lam * lm_logp((u, v), w, mu, nu)
                    if sc > new.get((v, w), -1e18): new[(v, w)] = sc; bp[(v, w)] = (u, v)
            states, back = new, back + [bp]
        st = max(states, key=states.get); out = [st[1], st[0]]
        for bp in reversed(back): st = bp[st]; out.append(st[0])
        out = out[::-1]; return out[1:] if out[0] == V.BOS else out

    def evaluate(data, **kw):
        ok = tot = sent = 0
        for r, n in data:
            h = decode(r, **kw); tot += len(n); ok += sum(a == b for a, b in zip(h, n)); sent += h == n
        return ok / tot, sent / len(data)

    res = dict(oracle_recall_dev=oracle(dev), oracle_recall_test=oracle(te)); print("oracle candidate recall:", res, flush=True)
    grid = {}
    for order in (2, 3):
        for lam in (0.2, 0.35, 0.5):
            for mu in (1.0, 0.7, 0.4):
                for nu in ((0.0,) if order == 2 else (0.3, 0.6)):
                    acc, sacc = evaluate(dev[:1000], lam=lam, mu=mu, nu=nu, order=order); grid[f"order={order} lam={lam} mu={mu} nu={nu}"] = dict(token_acc=acc, sentence_acc=sacc)
                    print(f"  dev order={order} lam={lam} mu={mu} nu={nu}: token {acc:.4f} sent {sacc:.4f}", flush=True)
    res["dev_grid"] = grid; best = max(grid, key=lambda k: grid[k]["token_acc"]); res["best_dev_config"] = best
    kv = dict(x.split("=") for x in best.split()); kw = dict(order=int(kv["order"]), lam=float(kv["lam"]), mu=float(kv["mu"]), nu=float(kv["nu"]))
    print("best dev config:", best, flush=True)
    for name, k in (("in-domain bigram only (lam=0.2, mu=1)", dict(order=2, lam=0.2, mu=1.0, nu=0.0)), ("best", kw)):
        acc, sacc = evaluate(te, **k); res[f"test | {name}"] = dict(token_acc=acc, sentence_acc=sacc); print(f"TEST {name}: token {acc:.4f} sentence {sacc:.4f}", flush=True)
    json.dump(res, open(f"{OUT}/e2_rerank.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
