"""E2+: context-aware normaliser for Romanised Assamese (noisy-channel Viterbi decoder).

Baseline (e2_romanisation.lookup_normaliser) picks each token's most frequent native reading in isolation.
Here each Roman token gets a candidate set (seen readings; char-n-gram nearest neighbours for unseen
tokens; the token itself, since ~18% of tokens are Latin words that stay Roman) and the sentence is
decoded with a native-word bigram language model: argmax over sequences of
    sum_t [ log P(native_t | roman_t) + lambda * log P(native_t | native_{t-1}) ]
(a direct model plus LM rescoring; lambda = 0 is exactly the lookup baseline).
Same 80/20 sentence split and seed as the baseline, so the token accuracies are directly comparable.
"""
import sys, os, json, math
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from e2_romanisation import load, PATH

OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
SEED, K_SEEN, K_NN, LAMBDA = 0, 6, 8, 1.0
BOS = "<s>"


class Decoder:
    def fit(self, pairs):
        self.reads = defaultdict(Counter); self.uni = Counter(); self.bi = defaultdict(Counter)
        for r, n in pairs:
            prev = BOS
            for a, b in zip(r, n):
                self.reads[a.lower()][b] += 1; self.uni[b] += 1; self.bi[prev][b] += 1; prev = b
            self.uni[BOS] += 1
        self.N = sum(self.uni.values()); self.V = len(self.uni)
        self.vocab = list(self.reads); self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3)).fit(self.vocab)
        self.nn = NearestNeighbors(n_neighbors=K_NN, metric="cosine").fit(self.vec.transform(self.vocab))
        return self

    def lm(self, prev, w):
        """Absolute-discounting interpolated bigram, log-prob."""
        d = 0.75; c = self.bi.get(prev); pu = (self.uni.get(w, 0) + 0.5) / (self.N + 0.5 * (self.V + 1))
        if not c: return math.log(pu)
        tot = sum(c.values()); lam = d * len(c) / tot
        return math.log(max(c.get(w, 0) - d, 0) / tot + lam * pu)

    def candidates(self, tok):
        """{native: log P(roman | native)} for a lower-cased Roman token."""
        t = tok.lower(); out = {}
        if t in self.reads:
            c = self.reads[t]; tot = sum(c.values())
            for b, k in c.most_common(K_SEEN):
                out[b] = math.log(k / tot)  # direct model P(n | r); lambda=0 reproduces the lookup baseline
            if tok not in out: out[tok] = math.log(0.5 / (tot + 1))
        else:
            _, ix = self.nn.kneighbors(self.vec.transform([t]))
            for rank, i in enumerate(ix[0]):
                c = self.reads[self.vocab[i]]; b, k = c.most_common(1)[0]
                out.setdefault(b, math.log(k / sum(c.values())) - 0.5 * rank)
            out.setdefault(tok, math.log(0.3))  # unseen tokens are often Latin words kept as-is
        return out

    def decode(self, toks):
        cands = [self.candidates(t) for t in toks]
        back, score = [], {BOS: 0.0}
        for cd in cands:
            new, bp = {}, {}
            for w, em in cd.items():
                best = max(score, key=lambda p: score[p] + LAMBDA * self.lm(p, w))
                new[w] = score[best] + LAMBDA * self.lm(best, w) + em; bp[w] = best
            score, back = new, back + [bp]
        w = max(score, key=score.get); out = [w]
        for bp in reversed(back[1:]): w = bp[w]; out.append(w)
        return out[::-1]


def evaluate(dec, test, table):
    tot = ok = base_ok = iv = iv_ok = uv = uv_ok = sent_ok = 0
    for r, n in test:
        pred = dec.decode(r); allok = True
        for a, b, p in zip(r, n, pred):
            hit = p == b; tot += 1; ok += hit; allok &= hit; base_ok += table.get(a.lower()) == b
            if a.lower() in dec.reads: iv += 1; iv_ok += hit
            else: uv += 1; uv_ok += hit
        sent_ok += allok
    return dict(token_acc=ok / tot, in_vocab_acc=iv_ok / iv, unseen_acc=uv_ok / uv, sentence_acc=sent_ok / len(test),
                lookup_token_acc_same_split=base_ok / tot)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pairs, _ = load(PATH); rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(pairs)); cut = int(0.8 * len(pairs))
    tr, te = [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]
    dec = Decoder().fit(tr)
    table = {a: c.most_common(1)[0][0] for a, c in dec.reads.items()}
    res = {}
    for lam in (0.0, 0.2, 0.4, 0.7, 1.0):
        LAMBDA = lam; r = evaluate(dec, te[:2000], table); res[f"lambda={lam} (2k dev sentences)"] = r
        print(f"lambda {lam}: {r}", flush=True)
    LAMBDA = max((float(k.split("=")[1].split()[0]) for k in res), key=lambda l: res[f"lambda={l} (2k dev sentences)"]["token_acc"])
    res["final"] = evaluate(dec, te, table); res["final"]["lambda"] = LAMBDA
    print("full test set:", res["final"], flush=True)
    json.dump(res, open(f"{OUT}/e2_plus.json", "w"), indent=1, ensure_ascii=False)
