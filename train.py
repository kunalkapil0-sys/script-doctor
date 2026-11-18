"""Script Doctor: train a 'will this outperform the channel?' model from transcripts.

Label: a video is VIRAL if its age-adjusted views are in the top 25% of its own
channel, FLOP if in the bottom 25%. Middle videos are left out. Comparing a video
with its own channel removes channel size, so the model has to learn from the
script itself.
"""
import re, json, sys
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from scipy.sparse import hstack, csr_matrix

def num(s):
    if pd.isna(s): return np.nan
    s = str(s).replace(",", "").split()[0]
    mul = {"K": 1e3, "M": 1e6, "B": 1e9}
    return float(s[:-1]) * mul[s[-1]] if s[-1] in mul else float(s)

def age_years(s):
    if pd.isna(s): return np.nan
    m = re.match(r"(\d+)\s+(year|month|week|day)", str(s))
    if not m: return np.nan
    n, u = int(m[1]), m[2]
    return n * {"year": 1, "month": 1/12, "week": 1/52, "day": 1/365}[u]

def secs(s):
    try:
        p = [int(x) for x in str(s).split(":")]
        return sum(v * 60**i for i, v in enumerate(reversed(p)))
    except ValueError:
        return np.nan

def load():
    d = pd.read_csv("data/youtubers.csv")
    a = pd.DataFrame({"channel": d.Channel, "title": d.Title.fillna(""),
        "text": d.Transcript.fillna(""), "views": d.Views.map(num),
        "age": d.Released.map(age_years), "secs": d.Length.map(secs),
        "category": d.Category.fillna("")})
    m = pd.read_csv("data/mrbeast.csv")
    pub = pd.to_datetime(m.publish_date, errors="coerce")
    b = pd.DataFrame({"channel": "MrBeast", "title": m.title.fillna(""),
        "text": m.transcript.fillna(""), "views": m.views.astype(float),
        "age": (pd.Timestamp("2023-01-15") - pub).dt.days / 365,
        "secs": m.length.astype(float), "category": "Entertainment"})
    df = pd.concat([a, b], ignore_index=True)
    df = df[(df.text.str.split().str.len() > 50) & (df.views > 0)].reset_index(drop=True)
    return df

def label(df):
    df["lv"] = np.log10(df.views)
    df["la"] = np.log10(df.age.clip(lower=0.05))
    # within-channel age effect: older videos have had more time to collect views
    g = df.dropna(subset=["la"]).groupby("channel")
    x = g.la.transform(lambda v: v - v.mean()); y = g.lv.transform(lambda v: v - v.mean())
    slope = float((x * y).sum() / (x * x).sum())
    df["la_c"] = df.la - df.groupby("channel").la.transform("mean")
    df["score"] = df.lv - df.groupby("channel").lv.transform("mean") - slope * df.la_c.fillna(0)
    q = df.groupby("channel").score
    lo, hi = q.transform(lambda v: v.quantile(.25)), q.transform(lambda v: v.quantile(.75))
    df["y"] = np.where(df.score >= hi, 1, np.where(df.score <= lo, 0, -1))
    return df[df.y >= 0].reset_index(drop=True), slope

HOOK_WORDS = 120
def hook(t): return " ".join(t.split()[:HOOK_WORDS])

def feats(df):
    rows = []
    for t, ti, s in zip(df.text, df.title, df.secs):
        w = t.split(); n = len(w); h = w[:HOOK_WORDS]; hs = " ".join(h).lower()
        low = t.lower()
        sents = [x for x in re.split(r"[.!?]+", t) if x.strip()]
        rows.append({
            "words": np.log1p(n),
            "wpm": n / (s / 60) if s and s > 0 else 150,
            "minutes": (s or 0) / 60,
            "sent_len": n / max(len(sents), 1),
            "q_rate": t.count("?") / max(n, 1) * 100,
            "excl_rate": t.count("!") / max(n, 1) * 100,
            "you_rate": len(re.findall(r"\byou(r|'re)?\b", low)) / max(n, 1) * 100,
            "i_rate": len(re.findall(r"\b(i|i'm|me|my)\b", low)) / max(n, 1) * 100,
            "num_rate": len(re.findall(r"\d", t)) / max(n, 1) * 100,
            "diversity": len(set(x.lower() for x in w[:500])) / max(min(n, 500), 1),
            "hook_q": hs.count("?"), "hook_you": len(re.findall(r"\byou\b", hs)),
            "hook_num": len(re.findall(r"\d|hundred|thousand|million", hs)),
            "hook_stakes": len(re.findall(r"\b(never|worst|best|most|only|first|last|secret|crazy|insane|biggest|impossible|why|how|what)\b", hs)),
            "title_len": len(ti.split()), "title_caps": sum(c.isupper() for c in ti) / max(len(ti), 1),
            "title_num": int(bool(re.search(r"\d", ti))), "title_q": int("?" in ti),
            "title_dollar": int("$" in ti),
        })
    return pd.DataFrame(rows)

def model_matrix(tr, te, Ftr, Fte):
    v1 = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000, sublinear_tf=True, stop_words="english")
    v2 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True)
    v3 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True)
    mu, sd = Ftr.mean(), Ftr.std() + 1e-9
    Xtr = hstack([v1.fit_transform(tr.text), v2.fit_transform(tr.text.map(hook)), v3.fit_transform(tr.title),
                  csr_matrix(((Ftr - mu) / sd).values)]).tocsr()
    Xte = hstack([v1.transform(te.text), v2.transform(te.text.map(hook)), v3.transform(te.title),
                  csr_matrix(((Fte - mu) / sd).values)]).tocsr()
    return Xtr, Xte

def evaluate(df, F, splitter, groups=None, name=""):
    out = {}
    preds = {k: np.zeros(len(df)) for k in ["lr", "gb", "blend"]}
    for tr_i, te_i in splitter.split(df, df.y, groups):
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        Xtr, Xte = model_matrix(tr, te, F.iloc[tr_i], F.iloc[te_i])
        lr = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced").fit(Xtr, tr.y)
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced").fit(F.iloc[tr_i], tr.y)
        preds["lr"][te_i] = lr.predict_proba(Xte)[:, 1]
        preds["gb"][te_i] = gb.predict_proba(F.iloc[te_i])[:, 1]
    preds["blend"] = (preds["lr"] + preds["gb"]) / 2
    for k, p in preds.items():
        out[k] = {"balanced_acc": round(balanced_accuracy_score(df.y, p > .5), 3), "auc": round(roc_auc_score(df.y, p), 3)}
    # accuracy when the model is confident (top/bottom 20% of scores)
    p = preds["blend"]; lo, hi = np.quantile(p, [.2, .8]); m = (p <= lo) | (p >= hi)
    out["blend_confident_40pct"] = {"balanced_acc": round(balanced_accuracy_score(df.y[m], p[m] > np.median(p)), 3), "n": int(m.sum())}
    print(name, json.dumps(out, indent=1))
    return out, preds["blend"]

if __name__ == "__main__":
    df = load(); n_all = len(df)
    df, slope = label(df)
    F = feats(df)
    print(f"videos {n_all} -> labelled {len(df)} (viral {int(df.y.sum())}, flop {int((1-df.y).sum())}); age slope {slope:.2f}")
    r_new_creator, p = evaluate(df, F, GroupKFold(5), df.channel, "UNSEEN CHANNELS")
    r_known, _ = evaluate(df, F, StratifiedKFold(5, shuffle=True, random_state=0), None, "UNSEEN VIDEOS")
    # what separates viral from flop (for the analyzer's guidance)
    diff = (F[df.y == 1].median() - F[df.y == 0].median()).round(3)
    print("median feature gap viral-flop:\n", diff.to_string())
    json.dump({"n": len(df), "unseen_channels": r_new_creator, "unseen_videos": r_known,
               "feature_gap": diff.to_dict(), "viral_median": F[df.y == 1].median().round(3).to_dict(),
               "flop_median": F[df.y == 0].median().round(3).to_dict()}, open("results.json", "w"), indent=1)
