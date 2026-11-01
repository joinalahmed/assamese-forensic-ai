"""Shared evaluation: EER, Cllr, Cllr_min (PAV), logistic calibration, Tippett data.

Protocol (paper Sec. 4.6 / 5.1): scores are mapped to log-LRs by logistic
calibration fitted on trials from authors/speakers disjoint from the evaluation set.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def eer(ss, ds):
    """Equal error rate from same-source (ss) and different-source (ds) scores."""
    ss, ds = np.asarray(ss, float), np.asarray(ds, float)
    thr = np.sort(np.unique(np.concatenate([ss, ds])))
    miss = np.array([(ss < t).mean() for t in thr])
    fa = np.array([(ds >= t).mean() for t in thr])
    i = np.argmin(np.abs(miss - fa))
    return float((miss[i] + fa[i]) / 2)


def cllr(llr_ss, llr_ds):
    """Log-likelihood-ratio cost; llr arrays are natural-log LRs (Eq. 2 uses log2)."""
    ln2 = np.log(2)
    a = np.logaddexp(0, -np.asarray(llr_ss)) / ln2  # log2(1 + 1/LR)
    b = np.logaddexp(0, np.asarray(llr_ds)) / ln2   # log2(1 + LR)
    return float(0.5 * a.mean() + 0.5 * b.mean())


def cllr_min(llr_ss, llr_ds):
    """Cllr after optimal monotone calibration (pool-adjacent-violators)."""
    ss, ds = np.asarray(llr_ss, float), np.asarray(llr_ds, float)
    s = np.concatenate([ss, ds])
    y = np.concatenate([np.ones(len(ss)), np.zeros(len(ds))])
    # weight classes equally so the implied prior is 0.5
    w = np.concatenate([np.full(len(ss), 0.5 / len(ss)), np.full(len(ds), 0.5 / len(ds))])
    iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
    p = iso.fit(s, y, sample_weight=w).predict(s)
    llr = np.log(p) - np.log1p(-p)  # posterior odds at prior 0.5 == LR
    return cllr(llr[: len(ss)], llr[len(ss):])


class LogisticCalibrator:
    """log LR = a*score + b, fitted with class-balanced weights (prior 0.5)."""

    def fit(self, ss, ds):
        x = np.concatenate([ss, ds]).reshape(-1, 1)
        y = np.concatenate([np.ones(len(ss)), np.zeros(len(ds))])
        w = np.concatenate([np.full(len(ss), 0.5 / len(ss)), np.full(len(ds), 0.5 / len(ds))])
        self.m = LogisticRegression(C=1e6, max_iter=1000).fit(x, y, sample_weight=w * len(y))
        return self

    def __call__(self, s):
        return self.m.decision_function(np.asarray(s, float).reshape(-1, 1))


def report(ss_scores, ds_scores, cal):
    """Evaluate scores with a calibrator fitted elsewhere. Returns dict of paper metrics."""
    lss, lds = cal(ss_scores), cal(ds_scores)
    return dict(
        n_ss=len(ss_scores), n_ds=len(ds_scores),
        eer=eer(ss_scores, ds_scores),
        cllr=cllr(lss, lds), cllr_min=cllr_min(lss, lds),
        misleading_ss=float((lss / np.log(10) < -1).mean()),
        misleading_ds=float((lds / np.log(10) > 1).mean()),
    )


def tippett(llr_ss, llr_ds, ax, label_ss="same-source", label_ds="different-source"):
    lo = min(np.min(llr_ss), np.min(llr_ds)) / np.log(10)
    hi = max(np.max(llr_ss), np.max(llr_ds)) / np.log(10)
    xs = np.linspace(lo, hi, 400)
    for l, lab in ((llr_ss, label_ss), (llr_ds, label_ds)):
        l10 = np.asarray(l) / np.log(10)
        ax.plot(xs, [(l10 >= x).mean() for x in xs], label=lab)
    ax.set_xlabel("log10 LR"); ax.set_ylabel("proportion of trials with LR ≥ x"); ax.legend()
