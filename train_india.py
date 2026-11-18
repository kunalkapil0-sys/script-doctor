"""India packaging model: what title / tags / description patterns beat the same channel's other videos.

Source: Kaggle "YouTube Trending Video Dataset (updated daily)", IN_youtube_trending_data.csv.
Rows are daily snapshots, so each video keeps its last (highest) view count.
Label: top 25% vs bottom 25% of a channel's own trending videos, adjusted for how long
the video had been up. No transcripts exist in this data, so this scores packaging, not scripts.
"""
import re, json, sys
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from scipy.sparse import hstack, csr_matrix

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/IN_youtube_trending_data.csv"
CATS = sys.argv[2] if len(sys.argv) > 2 else "data/IN_category_id.json"
DEV = re.compile(r"[ऀ-ॿ]")
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")

def load():
    d = pd.read_csv(PATH, low_memory=False, encoding_errors="replace")
    if "publish_time" in d.columns:  # older datasnaek INvideos.csv layout
        d = d.rename(columns={"publish_time": "publishedAt", "channel_title": "channelId", "category_id": "categoryId", "views": "view_count"})
        d["trending_date"] = pd.to_datetime(d.trending_date, format="%y.%d.%m", errors="coerce").dt.tz_localize("UTC")
        d["tags"] = d.tags.astype(str).str.replace('"', "", regex=False)
    else:
        d["trending_date"] = pd.to_datetime(d.trending_date, utc=True, errors="coerce")
    d["publishedAt"] = pd.to_datetime(d.publishedAt, utc=True, errors="coerce")
    d["trending_date"] = d.trending_date + pd.Timedelta(hours=23)  # snapshot is end of day
    d = d.sort_values("view_count").groupby("video_id").tail(1)
    d["days"] = ((d.trending_date - d.publishedAt).dt.total_seconds() / 86400).clip(lower=0.25)
    try:
        cats = {int(i["id"]): i["snippet"]["title"] for i in json.load(open(CATS))["items"]}
    except Exception:
        cats = {}
    d["category"] = d.categoryId.map(cats).fillna(d.categoryId.astype(str))
    d = d[(d.view_count > 0) & d.days.notna()]
    d = d[d.groupby("channelId").video_id.transform("count") >= 8].reset_index(drop=True)
    for c in ["title", "tags", "description"]: d[c] = d[c].fillna("").astype(str)
    d["tags"] = d.tags.str.replace(r"^\[[Nn]one\]$", "", regex=True).str.replace("|", " ", regex=False)
    return d

def label(d):
    d["lv"] = np.log10(d.view_count); d["ld"] = np.log10(d.days)
    x = d.ld - d.groupby("channelId").ld.transform("mean"); y = d.lv - d.groupby("channelId").lv.transform("mean")
    slope = float((x * y).sum() / (x * x).sum())
    d["score"] = y - slope * x
    q = d.groupby("channelId").score
    lo, hi = q.transform(lambda v: v.quantile(.25)), q.transform(lambda v: v.quantile(.75))
    d["y"] = np.where(d.score >= hi, 1, np.where(d.score <= lo, 0, -1))
    return d, slope

def tfeats(d):
    t = d.title
    return pd.DataFrame({
        "title_words": t.str.split().str.len(),
        "title_chars": t.str.len(),
        "caps_ratio": t.map(lambda s: sum(c.isupper() for c in s) / max(sum(c.isalpha() for c in s), 1)),
        "hindi_script": t.map(lambda s: len(DEV.findall(s)) / max(len(s), 1)),
        "has_number": t.str.contains(r"\d").astype(int),
        "has_question": t.str.contains(r"\?").astype(int),
        "has_exclaim": t.str.contains("!").astype(int),
        "has_pipe": t.str.contains(r"\|").astype(int),
        "has_emoji": t.map(lambda s: int(bool(EMOJI.search(s)))),
        "has_brackets": t.str.contains(r"[\(\[]").astype(int),
        "has_shorts": (t + " " + d.tags).str.contains("#short", case=False).astype(int),
        "has_rupee": t.str.contains(r"₹|rs\.?\s?\d|rupee", case=False).astype(int),
        "official": t.str.contains("official", case=False).astype(int),
        "episode": t.str.contains(r"\bep(isode)?\.?\s?\d", case=False).astype(int),
        "n_tags": d.tags.str.split().str.len().fillna(0),
        "desc_len": np.log1p(d.description.str.len()),
    })

def mats(tr, te, Ftr, Fte):
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5, max_features=30000, sublinear_tf=True)
    vt = TfidfVectorizer(min_df=3, max_features=20000, sublinear_tf=True)
    vd = TfidfVectorizer(min_df=5, max_features=20000, sublinear_tf=True)
    mu, sd = Ftr.mean(), Ftr.std() + 1e-9
    A = [vw.fit_transform(tr.title), vc.fit_transform(tr.title), vt.fit_transform(tr.tags), vd.fit_transform(tr.description.str[:400]), csr_matrix(((Ftr - mu) / sd).values)]
    B = [vw.transform(te.title), vc.transform(te.title), vt.transform(te.tags), vd.transform(te.description.str[:400]), csr_matrix(((Fte - mu) / sd).values)]
    return hstack(A).tocsr(), hstack(B).tocsr()

def evaluate(d, F, splitter, groups, name):
    p = np.zeros(len(d))
    for a, b in splitter.split(d, d.y, groups):
        Xa, Xb = mats(d.iloc[a], d.iloc[b], F.iloc[a], F.iloc[b])
        p[b] = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(Xa, d.y.iloc[a]).predict_proba(Xb)[:, 1]
    lo, hi = np.quantile(p, [.2, .8]); m = (p <= lo) | (p >= hi)
    r = {"balanced_acc": round(balanced_accuracy_score(d.y, p > .5), 3), "auc": round(roc_auc_score(d.y, p), 3),
         "confident_40pct_acc": round(balanced_accuracy_score(d.y[m], p[m] > np.median(p)), 3)}
    print(name, r); return r

if __name__ == "__main__":
    raw = load(); n_videos, n_channels = len(raw), raw.channelId.nunique()
    d, slope = label(raw.copy())
    print(f"India: {n_videos} unique trending videos, {n_channels} channels (>=8 videos), dates {raw.trending_date.min().date()} to {raw.trending_date.max().date()}, age slope {slope:.2f}")
    # pattern lift on ALL labelled-score videos: within-channel score with vs without each title pattern
    Fall = tfeats(d)
    lift = {}
    for c in [c for c in Fall.columns if Fall[c].isin([0, 1]).all()]:
        on = d.score[Fall[c] == 1]; off = d.score[Fall[c] == 0]
        if len(on) >= 200: lift[c] = {"n": int(len(on)), "x_views_vs_channel": round(float(10 ** (on.mean() - off.mean())), 2)}
    print("title pattern lift:", json.dumps(lift, indent=1))
    lab = d[d.y >= 0].reset_index(drop=True)
    if len(lab) > 20000: lab = lab.groupby("y", group_keys=False).apply(lambda g: g.sample(10000, random_state=0)).reset_index(drop=True)
    F = tfeats(lab)
    print(f"labelled {len(lab)} (viral {int(lab.y.sum())}, flop {int((1 - lab.y).sum())})")
    r1 = evaluate(lab, F, GroupKFold(5), lab.channelId, "UNSEEN CHANNELS")
    r2 = evaluate(lab, F, StratifiedKFold(5, shuffle=True, random_state=0), None, "UNSEEN VIDEOS (same channels)")
    cat = raw.groupby("category").agg(videos=("video_id", "count"), median_views=("view_count", "median"), median_like_rate=("likes", lambda s: 0))
    cat["median_like_rate"] = raw.assign(lr=raw.likes / raw.view_count).groupby("category").lr.median().round(4)
    cat = cat[cat.videos >= 100].sort_values("median_views", ascending=False)
    print(cat.to_string())
    json.dump({"videos": n_videos, "channels": n_channels, "date_from": str(raw.trending_date.min().date()), "date_to": str(raw.trending_date.max().date()),
               "unseen_channels": r1, "unseen_videos": r2, "title_lift": lift,
               "categories": cat.reset_index().to_dict("records"),
               "median_title_words": float(raw.title.str.split().str.len().median()),
               "share_hindi_script_titles": round(float((raw.title.map(lambda s: bool(DEV.search(s)))).mean()), 3)},
              open("results_india.json", "w"), indent=1, default=float)
