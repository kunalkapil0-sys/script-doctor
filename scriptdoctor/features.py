"""Feature extraction shared by training and inference. Pure functions, no model state."""
import re
import numpy as np
import pandas as pd

DEV = re.compile(r"[ऀ-ॿ]")
EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")
SCENE = re.compile(r"\[[^\]]*\]|\([^)]*\)")  # [B-roll], (cut to) etc. are directions, not speech
DATES = re.compile(r"\b(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|june?|july?|aug(ust)?|sep(t(ember)?)?|"
                   r"oct(ober)?|nov(ember)?|dec(ember)?|20\d\d|\d{1,2}(st|nd|rd|th)?)\b", re.I)
STAKES = re.compile(r"\b(never|worst|best|most|only|first|last|secret|crazy|insane|biggest|impossible|why|how|what|"
                    r"nobody|everyone|truth|mistake|actually|kya|kaise|kyun|sabse|kabhi)\b", re.I)
TEASE = re.compile(r"teaser|promo|preview|releasing|coming soon|trailer", re.I)
TELLING = re.compile(r"\b(basically|so yeah|as you (can )?see|i think|kind of|sort of|you know|like i said|"
                     r"in this video|today we|today i|let me tell you|before we start|don'?t forget to)\b", re.I)
CTA = re.compile(r"\b(subscribe|follow|like (this|the) video|comment (below|down)|share (this|with)|link in (bio|description)|"
                 r"hit the bell)\b", re.I)

HOOK_WORDS = {"short": 8, "long": 75}       # ~3 s at 165 wpm vs ~30 s at 150 wpm
WPM = {"short": 165, "long": 150}


def spoken(text: str) -> str:
    """Script text with scene directions removed."""
    return re.sub(r"\s+", " ", SCENE.sub(" ", text or "")).strip()


def words(text: str):
    return re.findall(r"[\w'₹$]+", text or "")


def strip_dates(s: str) -> str:
    return DATES.sub(" ", s or "")


def script_features(text: str, seconds=None, fmt="long") -> dict:
    sp = spoken(text); w = words(sp); n = len(w)
    low = sp.lower(); hook = " ".join(w[:HOOK_WORDS.get(fmt, 75)]).lower()
    sents = [x for x in re.split(r"(?<=[.!?।])\s+", sp) if x.strip()]
    if not seconds or seconds <= 0:
        seconds = n / WPM.get(fmt, 150) * 60
    per = lambda k: k / max(n, 1) * 100
    return {
        "words": np.log1p(n),
        "wpm": n / (seconds / 60) if seconds else 150,
        "minutes": seconds / 60,
        "sent_len": n / max(len(sents), 1),
        "q_rate": per(sp.count("?")),
        "excl_rate": per(sp.count("!")),
        "you_rate": per(len(re.findall(r"\b(you|your|you're|aap|tum|tumhe|aapko)\b", low))),
        "i_rate": per(len(re.findall(r"\b(i|i'm|me|my|main|mera|mujhe)\b", low))),
        "num_rate": per(len(re.findall(r"\d", sp))),
        "diversity": len({x.lower() for x in w[:500]}) / max(min(n, 500), 1),
        "telling_rate": per(len(TELLING.findall(low))),
        "hook_q": hook.count("?"),
        "hook_you": len(re.findall(r"\b(you|aap|tum)\b", hook)),
        "hook_num": len(re.findall(r"\d|₹|\$|hundred|thousand|million|lakh|crore", hook)),
        "hook_stakes": len(STAKES.findall(hook)),
        "hindi_script": len(DEV.findall(sp)) / max(len(sp), 1),
    }


def title_features(title: str, tags: str = "") -> dict:
    t = title or ""
    return {
        "title_words": len(t.split()),
        "caps_ratio": sum(c.isupper() for c in t) / max(sum(c.isalpha() for c in t), 1),
        "t_hindi": len(DEV.findall(t)) / max(len(t), 1),
        "t_number": int(bool(re.search(r"\d", t))),
        "t_question": int("?" in t),
        "t_exclaim": int("!" in t),
        "t_pipe": int("|" in t),
        "t_emoji": int(bool(EMOJI.search(t))),
        "t_brackets": int(bool(re.search(r"[\(\[]", t))),
        "t_money": int(bool(re.search(r"₹|\$|rs\.?\s?\d|rupee|lakh|crore", t, re.I))),
        "t_tease": int(bool(TEASE.search(t))),
        "t_full": int(bool(re.search(r"\bfull\b|video song|official", t, re.I))),
        "n_tags": len((tags or "").split()),
    }


def frame(rows, fn):
    return pd.DataFrame([fn(*r) for r in rows])
