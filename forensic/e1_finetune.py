"""E1 neural: contrastive fine-tuning of a pretrained Indic encoder for Assamese authorship verification.

Analogue of what the pretrained speaker encoder did for E3. An XLM-R-style Indic encoder (IndicBERT v2 by
default; MuRIL via MODEL=...) is fine-tuned with a supervised InfoNCE loss on Wikipedia editor sessions: a batch
holds P editors x K sessions, every pair of sessions from the same editor is a positive, all others negatives.
Sessions are randomly cropped to 40-220 words during training so the embedding is robust to short questioned texts.

Editors are split by identity: the 150 evaluation editors are exactly those of e1_wiki.py (same seed), all
remaining editors with >=3 sessions are training data, so no evaluation editor is ever seen in fine-tuning.
Evaluation re-uses the author-disjoint two-fold profile protocol (cosine and S-norm on mean-pooled embeddings)
at 75-word and full-session query length, before and after fine-tuning, plus transfer to the Arambhani poets.
"""
import sys, os, json, math, time, random
from collections import defaultdict
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from metrics import LogisticCalibrator, report, tippett
from scipy.sparse import vstack as svstack
from e1_verification_plus import Fused
from e1_authorship import load_poems

SRC = "../data/new/aswiki/editor_sessions.jsonl"; OUT = sys.argv[2] if len(sys.argv) > 2 else "results"
MODEL = os.environ.get("MODEL", "../data/new/models/IndicBERTv2-MLM-only"); TAG = os.environ.get("TAG", "indicbert")
SEED, MIN_SESS, MIN_WORDS, MAX_AUTHORS, MAX_SESS, Q_WORDS = 0, 3, 40, 150, 12, 75
STEPS, P, K, MAXTOK, LR, TEMP = int(os.environ.get("STEPS", 1500)), int(os.environ.get("P", 16)), 2, 256, 2e-5, 0.05
CHUNK = 150  # words per chunk when embedding long texts (chunk embeddings are averaged)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); torch.manual_seed(SEED); random.seed(SEED)


def load_sessions():
    by = defaultdict(list)
    for l in open(SRC, encoding="utf-8"):
        r = json.loads(l)
        if len(r["text"].split()) >= MIN_WORDS: by[r["user"]].append(r["text"])
    return {a: v for a, v in by.items() if len(v) >= MIN_SESS}


class Encoder(nn.Module):
    def __init__(self, path):
        super().__init__(); self.tok = AutoTokenizer.from_pretrained(path); self.m = AutoModel.from_pretrained(path)
    def forward(self, texts):
        b = self.tok(texts, padding="max_length", truncation=True, max_length=MAXTOK, return_tensors="pt").to(DEV)  # fixed shape: varying shapes make the MPS allocator grow without bound
        h = self.m(**b).last_hidden_state; mask = b["attention_mask"].unsqueeze(-1).float()
        return F.normalize((h * mask).sum(1) / mask.sum(1), dim=-1)

    @torch.no_grad()
    def embed(self, texts, bs=32):
        """Long texts are split into CHUNK-word pieces, embedded separately and averaged, so a full session uses all its text."""
        self.eval(); pieces, owner = [], []
        for i, t in enumerate(texts):
            w = t.split(); ch = [" ".join(w[j:j + CHUNK]) for j in range(0, max(len(w), 1), CHUNK)] or [t]; pieces += ch; owner += [i] * len(ch)
        out = []
        for i in range(0, len(pieces), bs): out.append(self(pieces[i:i + bs]).float().cpu())
        Z = torch.cat(out).numpy(); E = np.zeros((len(texts), Z.shape[1]))
        for z, o in zip(Z, owner): E[o] += z
        return E / np.linalg.norm(E, axis=1, keepdims=True)


def crop(t):
    w = t.split(); n = random.randint(40, 220)
    if len(w) <= n: return t
    s = random.randint(0, len(w) - n); return " ".join(w[s:s + n])


def supcon(z, labels):
    sim = z @ z.T / TEMP; sim.fill_diagonal_(-1e4)
    pos = (labels[:, None] == labels[None, :]).float(); pos.fill_diagonal_(0)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(logp * pos).sum(1).div(pos.sum(1).clamp(min=1)).mean()


def finetune(enc, train_by):
    authors = [a for a in train_by if len(train_by[a]) >= K]; opt = torch.optim.AdamW(enc.parameters(), lr=LR, weight_decay=0.01)
    with torch.no_grad(): enc.eval(); enc(["warm-up"]); print("  warm-up forward ok", flush=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.1); enc.train(); t0 = time.time(); run = 0
    for step in range(1, STEPS + 1):
        batch_a = random.sample(authors, P); texts, labels = [], []
        for i, a in enumerate(batch_a):
            for t in random.sample(train_by[a], K): texts.append(crop(t)); labels.append(i)
        loss = supcon(enc(texts), torch.tensor(labels, device=DEV)); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(enc.parameters(), 1.0); opt.step(); sched.step(); run = 0.98 * run + 0.02 * loss.item() if run else loss.item()
        if step % 50 == 0: print(f"  step {step}/{STEPS} loss {run:.3f} ({time.time() - t0:.0f}s)", flush=True)


def verify(E, by_idx, authors, rng, snorm, X=None):
    """E: embedding matrix; by_idx[a] = (query row ids, known row ids) per session; profile protocol, two folds.
    If X (TF-IDF matrix, same rows) is given, returns the fusion of standardised neural and TF-IDF S-norm scores."""
    halves = [authors[::2], authors[1::2]]; folds = []
    for dev, ev in ((0, 1), (1, 0)):
        def scores(auth, pool, E=E, snorm=snorm):
            sparse = hasattr(E, "tocsr"); rows = lambda ids: E[ids]
            mean_unit = lambda M: (lambda v: v / (np.linalg.norm(v) + 1e-12))(np.asarray(M.mean(0)).ravel())
            Ppool = (svstack if sparse else np.vstack)([rows(by_idx[a][1]) for a in pool]); coh = np.vstack([mean_unit(rows(by_idx[a][1])) for a in pool])
            dot = (lambda M, v: np.asarray(M @ v).ravel())
            ss, ds, prof = [], [], {}
            for b in auth:
                p = mean_unit(rows(by_idx[b][1])); z = dot(Ppool, p); prof[b] = (p, z.mean(), z.std() + 1e-9)
            for a in auth:
                q_ids, k_ids = by_idx[a]
                for i in range(len(q_ids)):
                    q = np.asarray(rows([q_ids[i]]).todense()).ravel() if sparse else E[q_ids[i]]; t = coh @ q; tm, ts = t.mean(), t.std() + 1e-9
                    pk = mean_unit(rows([k for j, k in enumerate(k_ids) if j != i])); zk = dot(Ppool, pk)
                    s = float(q @ pk); ss.append(0.5 * ((s - zk.mean()) / (zk.std() + 1e-9) + (s - tm) / ts) if snorm else s)
                    for b in auth:
                        if b == a: continue
                        p, zm, zs = prof[b]; s = float(q @ p); ds.append(0.5 * ((s - zm) / zs + (s - tm) / ts) if snorm else s)
            return np.array(ss), np.array(ds)
        d1, d2 = halves[dev][::2], halves[dev][1::2]
        def both(auth, pool):
            a = scores(auth, pool)
            if X is None: return a
            b = scores(auth, pool, E=X, snorm=True); return a, b
        if X is None:
            sa, sb = both(d1, d2), both(d2, d1); d_ss, d_ds = np.r_[sa[0], sb[0]], np.r_[sa[1], sb[1]]; e_ss, e_ds = both(halves[ev], halves[dev])
        else:
            (sa, ta), (sb, tb) = both(d1, d2), both(d2, d1); (se, te_) = both(halves[ev], halves[dev])
            def std(pair_dev, pair_ev):
                allv = np.r_[pair_dev[0], pair_dev[1]]; m, sd = allv.mean(), allv.std() + 1e-9; return (pair_dev[0] - m) / sd, (pair_dev[1] - m) / sd, (pair_ev[0] - m) / sd, (pair_ev[1] - m) / sd
            n = std((np.r_[sa[0], sb[0]], np.r_[sa[1], sb[1]]), se); t = std((np.r_[ta[0], tb[0]], np.r_[ta[1], tb[1]]), te_)
            d_ss, d_ds, e_ss, e_ds = n[0] + t[0], n[1] + t[1], n[2] + t[2], n[3] + t[3]
        cal = LogisticCalibrator().fit(d_ss, d_ds); folds.append(report(e_ss, e_ds, cal))
    r = {k: float(np.mean([f[k] for f in folds])) for k in ("eer", "cllr", "cllr_min")}; r["n_ss"] = int(sum(f["n_ss"] for f in folds)); r["n_ds"] = int(sum(f["n_ds"] for f in folds)); return r


def evaluate(enc, eval_by, tag, res):
    for q_words in (Q_WORDS, None):
        texts, by_idx = [], {}
        for a, sess in eval_by.items():
            k_ids = list(range(len(texts), len(texts) + len(sess))); texts += sess
            q_ids = list(range(len(texts), len(texts) + len(sess))); texts += [" ".join(t.split()[:q_words]) if q_words else t for t in sess]; by_idx[a] = (q_ids, k_ids)
        E = enc.embed(texts); rng = np.random.RandomState(SEED)
        # TF-IDF features for score fusion: fitted on all evaluation texts (unsupervised, no author labels)
        X = Fused().fit(texts).transform(texts).tocsr()
        for sn, fuse in ((False, False), (True, False), (True, True)):
            r = verify(E, by_idx, sorted(eval_by), rng, sn, X if fuse else None)
            name = f"{tag} | query {'75 words' if q_words else 'full'} | {'fusion neural S-norm + TF-IDF S-norm' if fuse else ('cosine + S-norm' if sn else 'cosine')}"; res[name] = r
            print(f"  {name:90s} {r['n_ss']}/{r['n_ds']}  EER {r['eer']:.3f}  Cllr {r['cllr']:.3f}  Cllr_min {r['cllr_min']:.3f}", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); rng = np.random.RandomState(SEED)
    by = load_sessions(); authors = sorted(by); rng.shuffle(authors); eval_a = authors[:MAX_AUTHORS]; train_a = authors[MAX_AUTHORS:]
    eval_by = {a: [by[a][i] for i in rng.choice(len(by[a]), min(len(by[a]), MAX_SESS), replace=False)] for a in eval_a}  # identical to e1_wiki.py
    train_by = {a: by[a] for a in train_a}
    poems = load_poems("../data/poems/arambhani.txt"); pb = defaultdict(list)
    for p in poems: pb[p["author"]].append(p["text"])
    poets = {a: v for a, v in pb.items() if len(v) >= 3}
    print(f"device {DEV}  model {MODEL}  eval editors {len(eval_by)}  train editors {len(train_by)} ({sum(len(v) for v in train_by.values())} sessions)  poets {len(poets)}", flush=True)
    enc = Encoder(MODEL).to(DEV); res = {}
    if not os.environ.get("SKIP_ZEROSHOT"): print("zero-shot:", flush=True); evaluate(enc, eval_by, f"{TAG} zero-shot", res); evaluate(enc, poets, f"{TAG} zero-shot poets", res)
    print(f"fine-tuning {STEPS} steps (P={P} editors x K={K} sessions):", flush=True); finetune(enc, train_by)
    print("fine-tuned:", flush=True); evaluate(enc, eval_by, f"{TAG} fine-tuned", res); evaluate(enc, poets, f"{TAG} fine-tuned poets (transfer)", res)
    torch.save(enc.m.state_dict(), f"{OUT}/e1_{TAG}_finetuned.pt"); json.dump(res, open(f"{OUT}/e1_finetune_{TAG}.json", "w"), indent=1)
