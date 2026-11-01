"""E2 sentence-level: character transformer that back-transliterates a whole Romanised sentence at once.

Word-level systems (lookup 80%, Viterbi+LM 82%, char transformer per word 84%) are capped by in-vocabulary
ambiguity (one Roman form -> up to 31 native words). Only sentence context resolves that, so this model reads the
entire Roman sentence (characters) and writes the entire native sentence (characters), learning disambiguation,
"keep Latin" decisions and spacing jointly. Same 80/20 sentence split (seed 0) as every E2 script, 2,000 training
sentences held out as dev for checkpoint selection. Runs on Apple MPS if available.

Token accuracy is computed by whitespace-splitting the hypothesis and reference and aligning with difflib so that
merged/split words count as errors rather than shifting every later token.
"""
import sys, os, json, math, time, difflib
T0 = time.time()
import numpy as np, torch, torch.nn as nn
from e2_romanisation import load, PATH

OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
SEED = 0; EPOCHS = int(os.environ.get("EPOCHS", 10)); D = int(os.environ.get("D", 384)); LAYERS = int(os.environ.get("LAYERS", 4)); BATCH_CHARS = int(os.environ.get("BATCH_CHARS", 20000))  # token-budget batching: MPS is kernel-launch bound, so many short sentences per step
MAXLEN = 256; PAD, BOS, EOS, UNK = 0, 1, 2, 3
DEV = torch.device(os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")); torch.manual_seed(SEED); torch.set_num_threads(max(1, os.cpu_count() - 2))


class Vocab:
    def __init__(self, strings):
        self.itos = ["<pad>", "<s>", "</s>", "<unk>"] + sorted({c for s in strings for c in s}); self.stoi = {c: i for i, c in enumerate(self.itos)}
    def enc(self, s): return [self.stoi.get(c, UNK) for c in s[:MAXLEN - 2]]
    def dec(self, ids): return "".join(self.itos[i] for i in ids if i > UNK)


def pad(seqs, bos=False, eos=False):
    seqs = [([BOS] if bos else []) + s + ([EOS] if eos else []) for s in seqs]; L = -(-max(map(len, seqs)) // 16) * 16  # bucket to multiples of 16: few distinct shapes, so the MPS allocator reuses blocks
    return torch.tensor([s + [PAD] * (L - len(s)) for s in seqs], device=DEV)


class Seq2Seq(nn.Module):
    def __init__(self, nsrc, ntgt):
        super().__init__(); self.se, self.te = nn.Embedding(nsrc, D, padding_idx=PAD), nn.Embedding(ntgt, D, padding_idx=PAD); self.pos = nn.Embedding(MAXLEN + 2, D)
        self.tf = nn.Transformer(D, nhead=8, num_encoder_layers=LAYERS, num_decoder_layers=LAYERS, dim_feedforward=4 * D, dropout=0.1, batch_first=True, norm_first=True)
        self.out = nn.Linear(D, ntgt)
    def embed(self, e, x): return e(x) * math.sqrt(D) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
    def forward(self, src, tgt_in):
        mask = self.tf.generate_square_subsequent_mask(tgt_in.shape[1], device=src.device)
        h = self.tf(self.embed(self.se, src), self.embed(self.te, tgt_in), tgt_mask=mask, src_key_padding_mask=src == PAD, tgt_key_padding_mask=tgt_in == PAD, memory_key_padding_mask=src == PAD)
        return self.out(h)
    @torch.no_grad()
    def greedy(self, src, maxlen=None):
        """Fixed-shape greedy decoding: the decoder always runs over a pre-allocated (B, maxlen) target buffer, so
        every step has the same tensor shapes and the MPS caching allocator reuses blocks instead of growing (the
        growing-prefix variant leaked ~30 GB through cached per-shape allocations). The causal mask makes positions
        beyond the current step irrelevant."""
        maxlen = maxlen or min(MAXLEN, -(-(src.shape[1] + 12) // 16) * 16)
        self.eval(); mem = self.tf.encoder(self.embed(self.se, src), src_key_padding_mask=src == PAD)
        B = src.shape[0]; ys = torch.full((B, maxlen), PAD, device=src.device); ys[:, 0] = BOS
        done = torch.zeros(B, dtype=torch.bool, device=src.device); mask = self.tf.generate_square_subsequent_mask(maxlen, device=src.device)
        for t in range(1, maxlen):
            h = self.tf.decoder(self.embed(self.te, ys), mem, tgt_mask=mask, memory_key_padding_mask=src == PAD)
            nxt = self.out(h[:, t - 1]).argmax(-1); nxt[done] = PAD; ys[:, t] = nxt; done |= nxt == EOS
            if done.all(): break
        out = []
        for row in ys[:, 1:].tolist():
            out.append(row[:row.index(EOS)] if EOS in row else [i for i in row if i != PAD])
        return out


def token_acc(hyps, refs):
    """Aligned token accuracy: matched tokens / reference tokens, via difflib on the token sequences."""
    ok = tot = sent_ok = 0
    for h, r in zip(hyps, refs):
        ht, rt = h.split(), r.split(); tot += len(rt); sent_ok += ht == rt
        ok += sum(b.size for b in difflib.SequenceMatcher(a=rt, b=ht, autojunk=False).get_matching_blocks())
    return ok / tot, sent_ok / len(refs)


def translate(model, sv, tv, sents, bs=96):
    out = []
    for i in range(0, len(sents), bs):
        if DEV.type == "mps": torch.mps.empty_cache()
        chunk = sorted(range(i, min(i + bs, len(sents))), key=lambda k: len(sents[k]))
        if os.environ.get("VERBOSE"): print(f"    decode {i}/{len(sents)} ({time.time() - T0:.0f}s)", flush=True)
        ids = model.greedy(pad([sv.enc(sents[k]) for k in chunk])); dec = dict(zip(chunk, (tv.dec(x) for x in ids))); out += [dec[k] for k in range(i, min(i + bs, len(sents)))]
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pairs, _ = load(PATH); rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(pairs)); cut = int(0.8 * len(pairs)); tr, te = [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]
    tr = [(" ".join(r), " ".join(n)) for r, n in tr]; te = [(" ".join(r), " ".join(n)) for r, n in te]; dev, tr = tr[:2000], tr[2000:]
    sv, tv = Vocab([r for r, _ in tr]), Vocab([n for _, n in tr]); model = Seq2Seq(len(sv.itos), len(tv.itos)).to(DEV)
    print(f"device {DEV}  train {len(tr)}  dev {len(dev)}  test {len(te)}  params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M  src vocab {len(sv.itos)} tgt vocab {len(tv.itos)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), weight_decay=0.01); loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    sched = None  # created after batching below
    # fixed-shape batching for MPS: sentences are grouped by source-length bucket (multiples of 16) and every batch of a
    # bucket has the same size, so only ~16 distinct tensor shapes ever reach the GPU and the allocator reuses blocks
    buckets = {}
    for k in range(len(tr)): buckets.setdefault(-(-(min(len(tr[k][0]), MAXLEN) + 2) // 16) * 16, []).append(k)
    batches = []
    for L, ks in buckets.items():
        bs = max(8, BATCH_CHARS // L)
        batches += [ks[i:i + bs] for i in range(0, len(ks), bs)]
    print(f"{len(batches)} batches/epoch", flush=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EPOCHS * len(batches), pct_start=0.08)
    best, t0, hist = (0, None), time.time(), []
    for ep in range(EPOCHS):
        model.train(); rng.shuffle(batches); tot = 0
        for b in batches:
            src = pad([sv.enc(tr[k][0]) for k in b]); tgt = pad([tv.enc(tr[k][1]) for k in b], bos=True, eos=True)
            logits = model(src, tgt[:, :-1]); loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += loss.item() * len(b)
            if os.environ.get("VERBOSE") and batches.index(b) % 25 == 0: print(f"    batch {batches.index(b)}/{len(batches)} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
        if (ep + 1) % int(os.environ.get("DEV_EVERY", 1)) and ep + 1 != EPOCHS: print(f"epoch {ep + 1}/{EPOCHS} loss {tot / len(tr):.3f}  ({time.time() - t0:.0f}s)", flush=True); continue
        acc, sacc = token_acc(translate(model, sv, tv, [r for r, _ in dev[:300]]), [n for _, n in dev[:300]])  # 300-sentence dev subset for checkpoint selection; hist.append(dict(epoch=ep + 1, loss=tot / len(tr), dev_token_acc=acc, dev_sentence_acc=sacc))
        print(f"epoch {ep + 1}/{EPOCHS} loss {tot / len(tr):.3f}  dev token acc {acc:.4f}  sentence acc {sacc:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if acc > best[0]: best = (acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1]); torch.save(dict(model=best[1], sv=sv.itos, tv=tv.itos, D=D, LAYERS=LAYERS), f"{OUT}/e2_seq2seq.pt")
    hyps = translate(model, sv, tv, [r for r, _ in te]); acc, sacc = token_acc(hyps, [n for _, n in te])
    # in-vocab / unseen breakdown on aligned tokens, using the word-level training vocabulary of the same split
    seen = {a.lower() for r, _ in tr for a in r.split()}; iv = uv = iv_ok = uv_ok = 0
    for (r, n), h in zip(te, hyps):
        rt, nt, ht = r.split(), n.split(), h.split(); m = difflib.SequenceMatcher(a=nt, b=ht, autojunk=False)
        matched = set(); [matched.update(range(bl.a, bl.a + bl.size)) for bl in m.get_matching_blocks()]
        for i, a in enumerate(rt):
            if i >= len(nt): break
            if a.lower() in seen: iv += 1; iv_ok += i in matched
            else: uv += 1; uv_ok += i in matched
    res = dict(test_token_acc=acc, test_sentence_acc=sacc, in_vocab_acc=iv_ok / iv, unseen_acc=uv_ok / uv, best_dev_token_acc=best[0], history=hist, config=dict(D=D, LAYERS=LAYERS, EPOCHS=EPOCHS, BATCH_CHARS=BATCH_CHARS))
    print("TEST", {k: v for k, v in res.items() if k != "history"}, flush=True)
    for (r, n), h in list(zip(te, hyps))[:5]: print(" ROMAN:", r, "\n  REF :", n, "\n  HYP :", h, flush=True)
    json.dump(res, open(f"{OUT}/e2_seq2seq.json", "w"), indent=1, ensure_ascii=False)
