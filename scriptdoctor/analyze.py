"""Script Doctor inference: one call, one report.

    from scriptdoctor.analyze import ScriptDoctor
    sd = ScriptDoctor()                       # loads models/scriptdoctor.joblib
    report = sd.analyze(script, title="...", fmt="short", market="india")

What each number is:
  packaging   trained model (title + tags), tested at ~58% balanced accuracy on unseen Indian channels
  craft       rule engine: hook, pacing, clarity, payoff, ending. Expert heuristics, NOT validated on view data
  retention   simulated curve from the craft rules. An estimate, not measured audience data
  score       0.75 * craft + 0.25 * packaging percentile
The transcript model is loaded but given zero weight: on held-out channels it scored at chance (AUC 0.50).
"""
import os, re
import numpy as np, pandas as pd, joblib
from .features import (script_features, title_features, spoken, strip_dates, words, STAKES, TEASE, TELLING, CTA, HOOK_WORDS, WPM)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(HERE, "models", "scriptdoctor.joblib")
STOP = set("the a an and or but so to of in on at for with is are was were be been it this that i you he she they we my your "
           "me his her our their its there here just very really what who how when then than ".split())
REVEAL = re.compile(r"\b(secret|reason|turns out|the answer|finally|result|here'?s why|truth|revealed|isn'?t the|it'?s not the|"
                    r"the trick|asli|raaz|wajah)\b", re.I)
CURIOSITY = re.compile(r"\?|\b(but|until|turns out|secret|why|what happened|guess|wait|nobody|except|instead|however|lekin|par)\b|\d", re.I)
WEAK_OPEN = re.compile(r"^\W*(hi|hey|hello|namaste|what'?s up|welcome( back)?|guys|so today|today (we|i)|in this video|"
                       r"before we (start|begin)|aaj (hum|main))\b", re.I)


def _content(s):
    return {w.lower() for w in words(s) if len(w) > 2 and w.lower() not in STOP}


def _sentences(s):
    return [x.strip() for x in re.split(r"(?<=[.!?।])\s+", s) if x.strip()]


def _beats(script, target=7):
    """Paragraphs if the writer used them, otherwise even chunks of sentences."""
    paras = [p for p in re.split(r"\n\s*\n", script.strip()) if words(spoken(p))]
    if len(paras) >= 4:
        return paras[:12] if len(paras) <= 12 else _merge(paras, 12)
    sents = _sentences(spoken(script))
    if len(sents) <= target:
        return sents
    k = int(np.ceil(len(sents) / target))
    return [" ".join(sents[i:i + k]) for i in range(0, len(sents), k)]


def _merge(parts, n):
    k = int(np.ceil(len(parts) / n))
    return ["\n".join(parts[i:i + k]) for i in range(0, len(parts), k)]


def _pct(ref, p):
    return int(round(100 * np.searchsorted(ref, p) / max(len(ref), 1)))


def _clip(x, lo=0, hi=10):
    return float(max(lo, min(hi, x)))


class ScriptDoctor:
    def __init__(self, path=DEFAULT_MODEL):
        b = joblib.load(path)
        self.b, self.pack, self.metrics = b, b["packaging_model"], b["metrics"]

    # ---------- packaging (trained model) ----------
    def packaging(self, title, tags=""):
        if not title.strip():
            return None
        F = pd.DataFrame([title_features(title, tags)])
        F["title"] = [strip_dates(title)]; F["meta"] = [strip_dates(tags)]
        p = float(self.pack.predict_proba(F)[0, 1])
        return {"probability": round(p, 3), "percentile": _pct(self.b["ref_pack"], p),
                "tested_accuracy": self.metrics["packaging_india"]["balanced_accuracy"]}

    # ---------- craft + retention (rules) ----------
    def analyze(self, script, title="", fmt="short", market="india", tags=""):
        fmt = "short" if fmt.startswith("s") else "long"
        sp = spoken(script); w = words(sp); n = len(w)
        if n < 15:
            raise ValueError("Script is too short to analyse (need at least 15 spoken words).")
        f = script_features(script, None, fmt)
        seconds = n / WPM[fmt] * 60
        beats = _beats(script)
        hook_txt = " ".join(w[:HOOK_WORDS[fmt]]); first_sent = (_sentences(sp) or [sp])[0]

        # hook
        h = 4.0
        h += min(f["hook_stakes"], 2) * 1.2 + min(f["hook_num"], 2) * 1.0 + (1.0 if f["hook_q"] else 0) + (0.8 if f["hook_you"] else 0)
        first_len = len(words(first_sent))
        h += 1.0 if first_len <= (14 if fmt == "short" else 25) else -1.0
        weak_open = bool(WEAK_OPEN.search(sp))
        h -= 3.0 if weak_open else 0
        hook = _clip(h)

        # beat-level risks and the simulated curve
        lens = [len(words(spoken(b))) for b in beats]; med = float(np.median(lens)) or 1
        seen, rows, est = set(), [], 100.0
        reveal_pos = None; cum = 0
        base = 0.035 if fmt == "short" else 0.045
        for i, b in enumerate(beats):
            spb = spoken(b); cw = _content(spb); L = lens[i]
            pos = round(100 * (cum + L / 2) / n) if i else 0; cum += L
            rep = len(cw & seen) / max(len(cw), 1) if i else 0
            reasons, risk = [], 0.0
            if i and rep > 0.45: risk += 0.35; reasons.append("repeats earlier lines")
            t = len(TELLING.findall(spb))
            if t: risk += 0.25 * min(t, 2); reasons.append("filler or telling phrases")
            sl = [len(words(s)) for s in _sentences(spb)] or [L]
            if np.mean(sl) > 22: risk += 0.25; reasons.append("long sentences")
            if i and not CURIOSITY.search(spb): risk += 0.25; reasons.append("no new question, number or turn")
            if i and L > 1.7 * med and L > 25: risk += 0.25; reasons.append("long stretch without a cut")
            if i == 0 and hook < 6: risk += 0.4; reasons.append("weak opening")
            if i == len(beats) - 1 and CTA.search(spb) and not REVEAL.search(spb): risk += 0.15; reasons.append("generic call to action")
            if reveal_pos is None and i and REVEAL.search(spb): reveal_pos = pos
            if i:
                drop = base + 0.10 * risk + ((0.25 if fmt == "short" else 0.12) * (1 - hook / 10) if i == 1 else 0)
                est *= (1 - min(drop, 0.45))
            seen |= cw
            level = "high" if risk >= 0.5 else "med" if risk >= 0.25 else "low"
            label = " ".join(words(spb)[:6]) + ("…" if L > 6 else "")
            rows.append({"pos": pos, "label": label, "est": round(est), "risk": level, "reasons": reasons, "text": spb})
        avg_ret = round(float(np.mean([r["est"] for r in rows])))

        # payoff
        late = 55 if fmt == "short" else 80
        if reveal_pos is None: payoff = 5.0
        elif reveal_pos <= late: payoff = _clip(9 - max(0, reveal_pos - 30) / 10)
        else: payoff = _clip(7 - (reveal_pos - late) / 6)

        # pacing and clarity
        hi_risk = sum(r["risk"] == "high" for r in rows[1:]); med_risk = sum(r["risk"] == "med" for r in rows[1:])
        pacing = _clip(9 - 2.0 * hi_risk - 0.8 * med_risk - (1.5 if fmt == "short" and seconds > 75 else 0))
        clarity = _clip(9 - max(0, f["sent_len"] - 16) * 0.35 - f["telling_rate"] * 2)
        last = spoken(beats[-1])
        ending = _clip(7 + (1.5 if re.search(r"\d|tip|next|part 2|comment", last, re.I) else 0) - (2 if CTA.search(last) and not REVEAL.search(last) and not re.search(r"\bnext\b|part", last, re.I) else 0))
        retention = _clip(avg_ret / 10)
        scores = {"hook": round(hook, 1), "pacing": round(pacing, 1), "clarity": round(clarity, 1), "payoff": round(payoff, 1),
                  "ending": round(ending, 1), "retention": round(retention, 1)}
        craft = round(10 * (0.30 * hook + 0.20 * pacing + 0.15 * clarity + 0.15 * payoff + 0.05 * ending + 0.15 * retention))

        pk = self.packaging(title, tags)
        score = round(0.75 * craft + 0.25 * pk["percentile"]) if pk else craft
        verdict = ("Strong potential" if score >= 75 else "Promising" if score >= 60 else "Needs work" if score >= 45 else "Unlikely to hold viewers")

        fixes = self._fixes(script, sp, rows, f, fmt, hook, weak_open, first_sent, reveal_pos, late, seconds, title)
        return {
            "score": score, "verdict": verdict, "craft_score": craft, "packaging": pk, "scores": scores,
            "retention": [{k: r[k] for k in ("pos", "label", "est", "risk", "reasons")} for r in rows],
            "avg_retention_estimate": avg_ret, "runtime_seconds": round(seconds), "spoken_words": n,
            "hook": {"score": round(hook, 1), "text": hook_txt},
            "fixes": fixes, "keep": self._keep(rows, f, hook, payoff, title),
            "questions": self._questions(fixes, market),
            "honesty": {"packaging_tested_accuracy": self.metrics["packaging_india"]["balanced_accuracy"],
                        "transcript_model_tested_accuracy": self.metrics["script"]["balanced_accuracy"],
                        "note": "Craft and retention are rule-based estimates; only the packaging model is trained on view data. Not a view forecast."},
        }

    # ---------- feedback ----------
    def _best_line(self, sp):
        sents = _sentences(sp)[1:]
        if not sents: return None
        sc = [(len(STAKES.findall(s)) + len(re.findall(r"\d|₹|\$", s)) * .5 + ("?" in s) - len(words(s)) / 40, s) for s in sents]
        best = max(sc)
        return best[1] if best[0] > 0.5 else None

    def _fixes(self, script, sp, rows, f, fmt, hook, weak_open, first_sent, reveal_pos, late, seconds, title):
        out = []
        if weak_open:
            out.append({"priority": 1, "issue": "Opens with a greeting or intro", "quote": first_sent,
                        "suggestion": "Cut the greeting. Start on the most surprising thing in your script; say hello later, or never."})
        if hook < 6.5:
            best = self._best_line(sp)
            out.append({"priority": 1, "issue": "The hook doesn't create a question in the viewer's head", "quote": first_sent,
                        "suggestion": "Open with a specific claim, number or contrast that only the full video can explain."
                                      + (" Your strongest line is buried later. Try opening with it:" if best else ""),
                        "example": best})
        if reveal_pos is None:
            out.append({"priority": 2, "issue": "No clear payoff moment", "quote": None,
                        "suggestion": "Name the reveal out loud ('the reason is…', 'turns out…') so viewers know the promise was kept."})
        elif reveal_pos > late:
            r = next((x for x in rows if x["pos"] == reveal_pos), None)
            out.append({"priority": 2, "issue": f"The payoff arrives {reveal_pos}% of the way in", "quote": r and _sentences(r["text"])[0],
                        "suggestion": f"For a {'short' if fmt == 'short' else 'long video'}, tease it in the hook and land it by {late}% of the runtime."})
        risky = [r for r in rows[1:] if r["risk"] == "high" and not (r is rows[-1] and CTA.search(r["text"]))]
        if risky:
            worst = max(risky, key=lambda r: len(r["reasons"]))
            reasons = sorted({x for r in risky for x in r["reasons"]})
            out.append({"priority": 2, "issue": "Drop-off risk at " + ", ".join(f"{r['pos']}%" for r in risky) + ": " + ", ".join(reasons),
                        "quote": (_sentences(worst["text"]) or [worst["text"]])[0],
                        "suggestion": "Show it instead of saying it: cut to the visual, shorten each beat to one line, and end each beat with a new question or number that pulls into the next."})
        if f["telling_rate"] > 0.6:
            hit = TELLING.search(sp)
            out.append({"priority": 3, "issue": "Filler phrases slow the pace", "quote": hit and hit.group(0),
                        "suggestion": "Delete phrases like 'basically', 'in this video', 'let me tell you'. Each one is a second where viewers can leave."})
        if fmt == "short" and seconds > 75:
            out.append({"priority": 2, "issue": f"About {round(seconds)} s of speech is long for a short", "quote": None,
                        "suggestion": "Aim for 30–60 s. Cut the beat with the least new information."})
        if f["sent_len"] > 22:
            out.append({"priority": 3, "issue": "Sentences are long for spoken video", "quote": None,
                        "suggestion": "Break lines to under ~15 words; spoken scripts breathe in short sentences."})
        last = rows[-1]
        if CTA.search(last["text"]) and not re.search(r"\bnext\b|part|tip|\d", last["text"], re.I):
            out.append({"priority": 3, "issue": "Ending is a generic call to action", "quote": (_sentences(last["text"]) or [""])[-1],
                        "suggestion": "End on a useful tip or a hook for the next video; viewers share endings that give them something."})
        if title:
            if TEASE.search(title):
                out.append({"priority": 2, "issue": "Title reads like a teaser or promo", "quote": title,
                            "suggestion": "In 13,794 Indian trending videos, teaser/promo/preview titles got 0.85x the channel's usual views. Promise the full payoff instead."})
            tw = _content(title); hw = _content(" ".join(words(sp)[:60]))
            if tw and not (tw & hw):
                out.append({"priority": 3, "issue": "Title and opening don't match", "quote": title,
                            "suggestion": "Repeat the title's promise in the first line so viewers who clicked feel they're in the right place."})
            if len(title.split()) > 14:
                out.append({"priority": 3, "issue": "Title is long", "quote": title, "suggestion": "Front-load the hook; the first ~50 characters show in feeds."})
        out.sort(key=lambda x: x["priority"])
        return out[:6]

    def _keep(self, rows, f, hook, payoff, title):
        k = []
        if hook >= 7: k.append("Your opening already creates curiosity. Protect it in edits.")
        if f["num_rate"] > 1: k.append("Specific numbers make it believable.")
        if f["you_rate"] > 1.5: k.append("You talk to the viewer directly.")
        if payoff >= 8: k.append("The payoff lands early enough to reward the click.")
        lows = [r for r in rows[1:] if r["risk"] == "low"]
        if lows: k.append(f"The beat at {lows[0]['pos']}% (“{lows[0]['label']}”) moves well.")
        return k or ["There's a real idea here. Fix the structure before rewriting the words."]

    def _questions(self, fixes, market):
        q = ["What is the one moment in this video nobody else could film or say?",
             "If a viewer remembers one line tomorrow, which line should it be? Is it in the first 5 seconds?"]
        issues = " ".join(x["issue"] for x in fixes)
        if "payoff" in issues.lower(): q.append("What does the viewer get at the end, and can you hint at it in the first line?")
        if "Drop-off" in issues: q.append("Which beat could you show instead of explain?")
        if market == "india": q.append("Would this line sound natural in the language your audience speaks at home, Hinglish included?")
        return q[:4]
