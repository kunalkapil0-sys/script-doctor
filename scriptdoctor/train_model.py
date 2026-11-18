"""Train Script Doctor's two models and save them as one file: models/scriptdoctor.joblib

  script model     transcript -> P(beats the creator's own channel)   English data: YouTubers + MrBeast
  packaging model  title/tags -> P(beats the creator's own channel)   India trending + English titles

Run from the project folder:  .venv/bin/python -m scriptdoctor.train_model
"""
import os, sys, json, datetime
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, os.getcwd())
import train as EN                     # English transcript loading + within-channel labels
import train_india as IN               # India trending loading + within-channel labels
from scriptdoctor.features import script_features, title_features, spoken, strip_dates, HOOK_WORDS

OUT = "models/scriptdoctor.joblib"


def hook_text(texts):
    return [" ".join(spoken(t).split()[:HOOK_WORDS["long"]]) for t in texts]


# ---------- script model ----------
def script_frame(df):
    F = pd.DataFrame([script_features(t, s, "long") for t, s in zip(df.text, df.secs)])
    F["text"] = [spoken(t) for t in df.text]
    F["hook"] = hook_text(df.text)
    return F


def script_pipeline(num_cols):
    ct = ColumnTransformer([
        ("full", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000, sublinear_tf=True, stop_words="english"), "text"),
        ("hook", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True), "hook"),
        ("num", StandardScaler(), num_cols),
    ])
    return Pipeline([("x", ct), ("clf", LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced"))])


# ---------- packaging model ----------
def pack_frame(titles, tags, descs):
    F = pd.DataFrame([title_features(t, g) for t, g in zip(titles, tags)])
    F["title"] = [strip_dates(t) for t in titles]
    F["meta"] = [strip_dates(f"{g} {str(d)[:400]}") for g, d in zip(tags, descs)]
    return F


def pack_pipeline(num_cols):
    ct = ColumnTransformer([
        ("tw", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True), "title"),
        ("tc", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5, max_features=30000, sublinear_tf=True), "title"),
        ("meta", TfidfVectorizer(min_df=5, max_features=20000, sublinear_tf=True), "meta"),
        ("num", StandardScaler(), num_cols),
    ])
    return Pipeline([("x", ct), ("clf", LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced"))])


def cv_eval(make, X, y, groups, name):
    p = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, groups):
        p[b] = make().fit(X.iloc[a], y[a]).predict_proba(X.iloc[b])[:, 1]
    r = {"balanced_accuracy": round(float(balanced_accuracy_score(y, p > .5)), 3), "auc": round(float(roc_auc_score(y, p)), 3), "n": int(len(y))}
    lo, hi = np.quantile(p, [.2, .8]); m = (p <= lo) | (p >= hi)
    r["confident_40pct_accuracy"] = round(float(balanced_accuracy_score(y[m], p[m] > np.median(p))), 3)
    print(f"{name}: {r}")
    return r


def fit_final(make, X, y):
    return CalibratedClassifierCV(make(), method="sigmoid", cv=5).fit(X, y)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    # English transcripts
    en, _ = EN.label(EN.load())
    Xs = script_frame(en); ys = en.y.values
    s_num = [c for c in Xs.columns if c not in ("text", "hook")]
    s_metrics = cv_eval(lambda: script_pipeline(s_num), Xs, ys, en.channel.values, "script model (unseen channels)")

    # India trending titles + English titles
    IN.PATH, IN.CATS = "data/INvideos.csv", "data/IN_category_id.json"
    ind, _ = IN.label(IN.load()); ind = ind[ind.y >= 0]
    pk = pd.concat([
        pd.DataFrame({"title": ind.title, "tags": ind.tags, "desc": ind.description, "y": ind.y, "g": "IN:" + ind.channelId.astype(str), "mkt": "india"}),
        pd.DataFrame({"title": en.title, "tags": "", "desc": "", "y": en.y, "g": "EN:" + en.channel, "mkt": "global"}),
    ], ignore_index=True)
    Xp = pack_frame(pk.title, pk.tags, pk.desc); yp = pk.y.values
    p_num = [c for c in Xp.columns if c not in ("title", "meta")]
    p_metrics = cv_eval(lambda: pack_pipeline(p_num), Xp, yp, pk.g.values, "packaging model (unseen channels)")
    p_metrics_india = cv_eval(lambda: pack_pipeline(p_num), Xp[pk.mkt.eq("india").values].reset_index(drop=True),
                              yp[pk.mkt.eq("india").values], pk.g[pk.mkt.eq("india")].values, "packaging model, India only")

    print("fitting final calibrated models...")
    script_model = fit_final(lambda: script_pipeline(s_num), Xs, ys)
    pack_model = fit_final(lambda: pack_pipeline(p_num), Xp, yp)

    # reference distributions so a new prediction can be shown as a percentile
    ref_s = np.sort(script_model.predict_proba(Xs)[:, 1]); ref_p = np.sort(pack_model.predict_proba(Xp)[:, 1])
    # feature medians of hits vs flops, used by the rule engine as evidence
    Sf = Xs[s_num]
    bundle = {
        "version": "1.0", "trained": datetime.date.today().isoformat(),
        "script_model": script_model, "packaging_model": pack_model,
        "script_num_cols": s_num, "pack_num_cols": p_num,
        "ref_script": ref_s[:: max(1, len(ref_s) // 500)], "ref_pack": ref_p[:: max(1, len(ref_p) // 500)],
        "metrics": {"script": s_metrics, "packaging": p_metrics, "packaging_india": p_metrics_india},
        "hit_median": Sf[ys == 1].median().round(3).to_dict(), "flop_median": Sf[ys == 0].median().round(3).to_dict(),
        "data": {"english_transcripts": int(len(en)), "packaging_titles": int(len(pk)), "india_titles": int((pk.mkt == "india").sum())},
    }
    joblib.dump(bundle, OUT, compress=3)
    json.dump({k: bundle[k] for k in ("version", "trained", "metrics", "data")}, open("models/model_card.json", "w"), indent=1)
    print(f"saved {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
