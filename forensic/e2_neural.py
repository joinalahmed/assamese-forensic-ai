"""E2 neural: character-level transformer transliterator for Romanised Assamese words (PyTorch, CPU).

Trained on the word pairs of the 80% training sentences (same split/seed as e2_romanisation.py); the native
reading of each training Roman token is its majority reading, so the model learns the normalising mapping.
At test time it is used two ways:
  * as the unseen-token model (replacing the char-n-gram nearest-neighbour fallback), and
  * as an extra candidate (with its log-probability) inside the Viterbi decoder of e2_normaliser_plus.py.
A 2-layer d=192 transformer trains in ~10 min on CPU; set EPOCHS lower to go faster.
Round 8 flags: NFC=1 applies Unicode NFC to the native side; NODEV=1 drops the first 2,000 training sentences
(the dev slice used by e2_rerank2.py / e2_discrim.py) from the training pairs, so dev-tuned rerankers see
leak-free neural scores.
"""
import sys, os, json, math, time, unicodedata
from collections import Counter, defaultdict
import numpy as np, torch, torch.nn as nn
from e2_romanisation import load, PATH
import e2_normaliser_plus as V

OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
SEED, EPOCHS, BATCH, D, MAXLEN = 0, int(os.environ.get("EPOCHS", 6)), 256, int(os.environ.get("D", 192)), 24
BEAM, LAYERS, TAG = int(os.environ.get("BEAM", 1)), int(os.environ.get("LAYERS", 2)), os.environ.get("TAG", "")
torch.manual_seed(SEED); torch.set_num_threads(max(1, os.cpu_count() - 1))
PAD, BOS, EOS = 0, 1, 2


class Vocab:
    def __init__(self, strings):
        self.itos = ["<pad>", "<s>", "</s>", "<unk>"] + sorted({c for s in strings for c in s}); self.stoi = {c: i for i, c in enumerate(self.itos)}
    def enc(self, s): return [self.stoi.get(c, 3) for c in s[:MAXLEN]]
    def dec(self, ids): return "".join(self.itos[i] for i in ids if i > 3)


def pad(seqs, bos=False, eos=False):
    seqs = [([BOS] if bos else []) + s + ([EOS] if eos else []) for s in seqs]; L = max(map(len, seqs))
    return torch.tensor([s + [PAD] * (L - len(s)) for s in seqs])


class Translit(nn.Module):
    def __init__(self, nsrc, ntgt):
        super().__init__()
        self.se, self.te = nn.Embedding(nsrc, D, padding_idx=PAD), nn.Embedding(ntgt, D, padding_idx=PAD)
        self.pos = nn.Embedding(MAXLEN + 2, D)
        self.tf = nn.Transformer(D, nhead=4, num_encoder_layers=LAYERS, num_decoder_layers=LAYERS, dim_feedforward=4 * D, dropout=0.1, batch_first=True)
        self.out = nn.Linear(D, ntgt)

    def embed(self, e, x): return e(x) + self.pos(torch.arange(x.shape[1])[None])

    def forward(self, src, tgt_in):
        mask = self.tf.generate_square_subsequent_mask(tgt_in.shape[1])
        h = self.tf(self.embed(self.se, src), self.embed(self.te, tgt_in), tgt_mask=mask, src_key_padding_mask=src == PAD, tgt_key_padding_mask=tgt_in == PAD, memory_key_padding_mask=src == PAD)
        return self.out(h)

    @torch.no_grad()
    def greedy(self, src, maxlen=MAXLEN + 1):
        """Returns decoded id lists and total log-prob per item."""
        self.eval(); mem = self.tf.encoder(self.embed(self.se, src), src_key_padding_mask=src == PAD)
        ys = torch.full((src.shape[0], 1), BOS); lp = torch.zeros(src.shape[0]); done = torch.zeros(src.shape[0], dtype=torch.bool)
        for _ in range(maxlen):
            mask = self.tf.generate_square_subsequent_mask(ys.shape[1])
            h = self.tf.decoder(self.embed(self.te, ys), mem, tgt_mask=mask, memory_key_padding_mask=src == PAD)
            logp = torch.log_softmax(self.out(h[:, -1]), -1); nxt = logp.argmax(-1)
            lp += torch.where(done, torch.zeros(()), logp.gather(1, nxt[:, None])[:, 0]); ys = torch.cat([ys, nxt[:, None]], 1); done |= nxt == EOS
            if done.all(): break
        return [[i for i in row.tolist()[1:] if i not in (PAD,)][: (row.tolist()[1:] + [EOS]).index(EOS)] for row in ys], lp


@torch.no_grad()
def beam(model, src, k=BEAM, maxlen=MAXLEN + 1):
    """Per item: list of (ids, logprob) for the k best hypotheses. Batched over items, k hypotheses each."""
    model.eval(); n = src.shape[0]; mem = model.tf.encoder(model.embed(model.se, src), src_key_padding_mask=src == PAD)
    mem = mem.repeat_interleave(k, 0); srck = src.repeat_interleave(k, 0)
    ys = torch.full((n * k, 1), BOS); lp = torch.zeros(n, k); lp[:, 1:] = -1e9; done = torch.zeros(n * k, dtype=torch.bool)
    for _ in range(maxlen):
        mask = model.tf.generate_square_subsequent_mask(ys.shape[1])
        h = model.tf.decoder(model.embed(model.te, ys), mem, tgt_mask=mask, memory_key_padding_mask=srck == PAD)
        logp = torch.log_softmax(model.out(h[:, -1]), -1); V = logp.shape[-1]
        logp[done] = -1e9; logp[done, PAD] = 0  # finished hypotheses only extend with PAD at no cost
        cand = (lp.reshape(-1, 1) + logp).reshape(n, k * V); top, idx = cand.topk(k, -1)
        src_hyp, tok = idx // V, idx % V; base = (torch.arange(n) * k)[:, None]
        ys = torch.cat([ys[(base + src_hyp).reshape(-1)], tok.reshape(-1, 1)], 1); done = done[(base + src_hyp).reshape(-1)] | (tok.reshape(-1) == EOS); lp = top
        if done.all(): break
    out = []
    for i in range(n):
        hyps = []
        for j in range(k):
            row = ys[i * k + j].tolist()[1:]; row = row[:row.index(EOS)] if EOS in row else row; hyps.append(([t for t in row if t != PAD], float(lp[i, j])))
        out.append(hyps)
    return out


def train(model, pairs, sv, tv):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01); loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.05)
    steps = EPOCHS * math.ceil(len(pairs) / BATCH); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    rng = np.random.RandomState(SEED); t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = rng.permutation(len(pairs)); tot = 0
        for b in range(0, len(pairs), BATCH):
            batch = [pairs[i] for i in perm[b:b + BATCH]]
            src = pad([sv.enc(r) for r, _ in batch]); tgt = pad([tv.enc(n) for _, n in batch], bos=True, eos=True)
            logits = model(src, tgt[:, :-1]); loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item() * len(batch)
        print(f"epoch {ep + 1}/{EPOCHS} loss {tot / len(pairs):.3f}  ({time.time() - t0:.0f}s)", flush=True)


def predict(model, sv, tv, words, bs=512):
    """Top-1 strings and log-probs; with BEAM>1 also returns per-word lists of (string, logprob) alternatives."""
    out, lps, alts = [], [], []
    for b in range(0, len(words), bs):
        src = pad([sv.enc(w) for w in words[b:b + bs]])
        if BEAM > 1:
            for hyps in beam(model, src): out.append(tv.dec(hyps[0][0])); lps.append(hyps[0][1]); alts.append([(tv.dec(h), l) for h, l in hyps])
        else:
            ids, lp = model.greedy(src); out += [tv.dec(i) for i in ids]; lps += lp.tolist(); alts += [[(tv.dec(i), l)] for i, l in zip(ids, lp.tolist())]
    predict.alts = alts; return out, lps


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pairs, _ = load(PATH); rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(pairs)); cut = int(0.8 * len(pairs)); tr, te = [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]
    if os.environ.get("NFC"):
        nfc = lambda t: unicodedata.normalize("NFC", t); tr = [(r, [nfc(x) for x in n]) for r, n in tr]; te = [(r, [nfc(x) for x in n]) for r, n in te]; print("NFC-normalised native side", flush=True)
    tr_fit = tr[2000:] if os.environ.get("NODEV") else tr
    if os.environ.get("NODEV"): print("NODEV: training on the 80% split minus its first 2,000 (dev) sentences", flush=True)
    reads = defaultdict(Counter)
    for r, n in tr_fit:
        for a, b in zip(r, n): reads[a.lower()][b] += 1
    # one training example per Roman type, weighted by frequency through repetition (capped), target = majority reading
    wp = []
    for a, c in reads.items():
        b, k = c.most_common(1)[0]; wp += [(a, b)] * min(1 + int(math.log2(k)), 6)
    print(f"{len(reads)} Roman types → {len(wp)} weighted training pairs", flush=True)
    sv, tv = Vocab([a for a, _ in wp]), Vocab([b for _, b in wp]); model = Translit(len(sv.itos), len(tv.itos))
    train(model, wp, sv, tv)
    torch.save(dict(model=model.state_dict(), sv=sv.itos, tv=tv.itos), f"{OUT}/e2_translit{TAG}.pt")

    # 1) unseen-token accuracy: neural vs nearest-neighbour fallback vs copy
    unseen = [(a, b) for r, n in te for a, b in zip(r, n) if a.lower() not in reads]
    uw = [a.lower() for a, _ in unseen]; pred, lp = predict(model, sv, tv, uw)
    res = dict(unseen_tokens=len(unseen), neural_unseen_acc=float(np.mean([p == b for p, (_, b) in zip(pred, unseen)])),
               copy_unseen_acc=float(np.mean([a == b for a, b in unseen])))
    print("unseen:", res, flush=True)

    # 2) plug into the Viterbi decoder as an extra candidate for every token (lambda from e2_normaliser_plus)
    dec = V.Decoder().fit(tr_fit); table = {a: c.most_common(1)[0][0] for a, c in dec.reads.items()}
    all_toks = sorted({a.lower() for r, _ in te for a in r}); npred, nlp = predict(model, sv, tv, all_toks); neural = dict(zip(all_toks, predict.alts))
    base_cand = dec.candidates

    def cands(tok):
        out = base_cand(tok); hyps = neural[tok.lower()]; top = hyps[0][1]
        if tok.lower() not in dec.reads:
            out = {k: v for k, v in out.items() if k == tok}  # replace NN fallback by the neural readings (keep "copy")
            for p, lp_ in hyps: out[p] = max(out.get(p, -1e9), (lp_ - top) * 0.5 + top * 0.3 + math.log(0.6))
            out[tok] = out.get(tok, math.log(0.3)) - 0.5
        else:
            for p, lp_ in hyps: out[p] = max(out.get(p, -1e9), (lp_ - top) * 0.5 + top * 0.3 + math.log(0.15))  # weak extra candidates
        return out

    dec.candidates = cands
    res["config"] = dict(epochs=EPOCHS, d=D, layers=LAYERS, beam=BEAM)
    for lam in (0.2, 0.4):
        V.LAMBDA = lam; r = V.evaluate(dec, te, table); res[f"viterbi+neural lambda={lam}"] = r; print(f"viterbi+neural lambda={lam}:", r, flush=True)
    json.dump(res, open(f"{OUT}/e2_neural{TAG}.json", "w"), indent=1, ensure_ascii=False)
