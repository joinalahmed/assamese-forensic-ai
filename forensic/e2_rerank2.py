"""E2 rerank, stage 2: k-best lattice paths rescored by the sentence-level seq2seq model.

e2_rerank.py showed the word-level lattice contains the reference token 96.5% of the time but the decoder picks it
84.1%; a 113M-token news LM added only +0.2. The missing information is the *whole Roman sentence*, which only the
sentence-level char transformer (e2_seq2seq.py, 67% alone) conditions on. So: beam search over the lattice with the
best bigram configuration to get the k best native sentences per input, score each with the seq2seq model's
teacher-forced log-likelihood given the Roman sentence, and pick argmax of  lattice_score + beta * seq2seq_logprob.
beta is tuned on 1,000 dev sentences with the word-level decoder refitted *without* those sentences (the earlier
dev numbers were inflated by that leak); the test split is unchanged.
"""
import os, sys, json, math, time, pickle, unicodedata
import numpy as np
os.environ.setdefault("D", "256"); os.environ.setdefault("LAYERS", "2"); os.environ.setdefault("BEAM", "5")
import torch
import e2_neural as N
import e2_normaliser_plus as V
from e2_romanisation import load, PATH
from e2_rerank import KN, build_big_lm, NATIVE, K_SEEN

OUT = sys.argv[2] if len(sys.argv) > 2 else "results"; SEED, KBEST, LAM, MU = 0, 10, 0.2, 0.7
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_seq2seq():
    ck = torch.load(f"{OUT}/e2_seq2seq.pt", map_location="cpu")
    os.environ["D"], os.environ["LAYERS"] = str(ck["D"]), str(ck["LAYERS"])
    import importlib, e2_seq2seq as S; importlib.reload(S)
    sv, tv = S.Vocab([]), S.Vocab([]); sv.itos, tv.itos = ck["sv"], ck["tv"]; sv.stoi, tv.stoi = {c: i for i, c in enumerate(sv.itos)}, {c: i for i, c in enumerate(tv.itos)}
    m = S.Seq2Seq(len(sv.itos), len(tv.itos)); m.load_state_dict(ck["model"]); m.to(DEV).eval(); return S, m, sv, tv


@torch.no_grad()
def s2s_logprob(S, m, sv, tv, pairs, bs=128):
    """Teacher-forced log P(native | roman) summed over target characters, for a list of (roman, native) strings."""
    out = np.zeros(len(pairs)); order = sorted(range(len(pairs)), key=lambda k: (len(pairs[k][0]), len(pairs[k][1])))
    for i in range(0, len(order), bs):
        ix = order[i:i + bs]
        srcs = [sv.enc(pairs[k][0]) for k in ix]; tgts = [[S.BOS] + tv.enc(pairs[k][1]) + [S.EOS] for k in ix]
        Ls, Lt = -(-max(map(len, srcs)) // 16) * 16, -(-max(map(len, tgts)) // 16) * 16
        src = torch.tensor([s + [S.PAD] * (Ls - len(s)) for s in srcs], device=DEV); tgt = torch.tensor([t + [S.PAD] * (Lt - len(t)) for t in tgts], device=DEV)
        logits = m(src, tgt[:, :-1]); lp = torch.log_softmax(logits.float(), -1).gather(-1, tgt[:, 1:, None])[..., 0]
        lp = (lp * (tgt[:, 1:] != S.PAD)).sum(1).cpu().numpy()
        for k, v in zip(ix, lp): out[k] = v
        if DEV.type == "mps": torch.mps.empty_cache()
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    pairs, _ = load(PATH); rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(pairs)); cut = int(0.8 * len(pairs)); tr, te = [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]
    if os.environ.get("NFC"):  # Unicode NFC on the native side: the corpus mixes precomposed য়/ড়/ঢ় (U+09DF/DC/DD) with base+nukta
        nfc = lambda t: unicodedata.normalize("NFC", t); tr = [(r, [nfc(x) for x in n]) for r, n in tr]; te = [(r, [nfc(x) for x in n]) for r, n in te]; print("NFC-normalised native side", flush=True)
    dev, tr_lm = tr[:2000], tr[2000:]; dev = dev[:1000]
    vocab = {}; big, vocab = build_big_lm(vocab)
    ck = torch.load(f"{OUT}/e2_translit_big.pt", map_location="cpu"); sv, tv = N.Vocab([]), N.Vocab([]); sv.itos, tv.itos = ck["sv"], ck["tv"]
    sv.stoi, tv.stoi = {c: i for i, c in enumerate(sv.itos)}, {c: i for i, c in enumerate(tv.itos)}; model = N.Translit(len(sv.itos), len(tv.itos)); model.load_state_dict(ck["model"]); model.eval()
    toks = sorted({a.lower() for r, _ in dev + te for a in r}); N.predict(model, sv, tv, toks); neural = dict(zip(toks, N.predict.alts))
    S, s2s, ssv, stv = load_seq2seq(); print("models loaded", flush=True)

    def make_system(train_pairs):
        dec = V.Decoder().fit(train_pairs); ind2 = KN(2)
        for _, n in train_pairs: ind2.add([vocab.setdefault(t, len(vocab)) for t in n])
        ind2.finish()
        def cands(tok):
            t = tok.lower(); out = {}
            if t in dec.reads:
                c = dec.reads[t]; tot = sum(c.values())
                for b, k in c.most_common(K_SEEN): out[b] = math.log(k / tot)
                out.setdefault(tok, math.log(0.5 / (tot + 1)))
            hyps = neural[t]; top = hyps[0][1]
            for p, lp in hyps:
                if os.environ.get("NFC"): p = unicodedata.normalize("NFC", p)
                out[p] = max(out.get(p, -1e9), (lp - top) * 0.5 + top * 0.3 + math.log(0.6 if t not in dec.reads else 0.15))
            if t not in dec.reads: out[tok] = max(out.get(tok, -1e9), math.log(0.3) - 0.5)
            return out
        def lm(h, w):
            wid, hid = vocab.get(w, -3), (vocab.get(h, -3),); return math.log(MU * ind2.prob(hid, wid) + (1 - MU) * big.prob(hid, wid))
        def kbest(toks):
            beams = [(0.0, [V.BOS])]
            for cd in [cands(t) for t in toks]:
                nxt = [(s + em + LAM * lm(path[-1], w), path + [w]) for s, path in beams for w, em in cd.items()]
                nxt.sort(key=lambda x: -x[0]); beams = nxt[:KBEST]
            return [(s, p[1:]) for s, p in beams]
        return kbest

    def run(data, kbest, betas):
        allk = [kbest(r) for r, _ in data]
        flat = [(" ".join(r), " ".join(h)) for (r, _), ks in zip(data, allk) for _, h in ks]; t0 = time.time(); lp = s2s_logprob(S, s2s, ssv, stv, flat); print(f"  scored {len(flat)} hypotheses ({time.time() - t0:.0f}s)", flush=True)
        res = {}; i = 0; per = []
        for (r, n), ks in zip(data, allk):
            per.append([(s, h, lp[i + j]) for j, (s, h) in enumerate(ks)]); i += len(ks)
        oracle = np.mean([any(h == n for _, h, _ in ks) for (r, n), ks in zip(data, per)])
        for beta in betas:
            ok = tot = sent = 0
            for (r, n), ks in zip(data, per):
                h = max(ks, key=lambda x: x[0] + beta * x[2])[1]; tot += len(n); ok += sum(a == b for a, b in zip(h, n)); sent += h == n
            res[beta] = dict(token_acc=ok / tot, sentence_acc=sent / len(data))
        return res, float(oracle)

    print("dev (decoder refitted without dev sentences):", flush=True)
    dres, doracle = run(dev, make_system(tr_lm), [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]); print(f"  {KBEST}-best oracle sentence acc {doracle:.3f}")
    for b, r in dres.items(): print(f"  beta={b}: token {r['token_acc']:.4f} sent {r['sentence_acc']:.4f}", flush=True)
    beta = max(dres, key=lambda b: dres[b]["token_acc"]); print("best beta", beta, flush=True)
    print("test:", flush=True)
    tres, toracle = run(te, make_system(tr), [0.0, beta]); print(f"  {KBEST}-best oracle sentence acc {toracle:.3f}")
    for b, r in tres.items(): print(f"  TEST beta={b}: token {r['token_acc']:.4f} sent {r['sentence_acc']:.4f}", flush=True)
    json.dump(dict(dev={str(k): v for k, v in dres.items()}, dev_oracle=doracle, best_beta=beta, test={str(k): v for k, v in tres.items()}, test_oracle=toracle), open(f"{OUT}/e2_rerank2{'_nfc' if os.environ.get('NFC') else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
