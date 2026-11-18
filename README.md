# Script Doctor model

A model that reads a video script and its title. It returns a score, an estimated retention curve, prioritized fixes, and questions that push the creator's own thinking. It runs offline, with no AI service needed.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Analyse a script (plain text; `[scene notes]` in brackets are ignored as speech):

```bash
.venv/bin/python -m scriptdoctor samples/vada_pav.txt --title "40 people wait 2 hours for this ₹10 vada pav" --format short --market india
```

Add `--json` for machine-readable output.

Run the web page and API together:

```bash
.venv/bin/python -m scriptdoctor --serve 8000
```

Then open http://127.0.0.1:8000 in your browser. The page (`scriptdoctor/static/index.html`) sends your script to the model and draws the score, retention curve, fixes and India benchmarks. For the API, `POST http://127.0.0.1:8000/analyze` with `{"script": "...", "title": "...", "format": "short|long", "market": "india|global"}`. `GET /health` returns the model's test metrics.

From Python:

```python
from scriptdoctor.analyze import ScriptDoctor
report = ScriptDoctor().analyze(script_text, title="...", fmt="short", market="india")
```

## What's inside `models/scriptdoctor.joblib`

| Part | What it does | How it was built | Tested accuracy (unseen channels) |
|---|---|---|---|
| Packaging model | Scores the title (plus tags, if given): does it look like the channel's hits or its flops? | Logistic regression on word and character n-grams plus title features. Trained on 7,286 Indian trending videos and 1,499 English videos, labelled top vs bottom 25% of their own channel's views and adjusted for age. Month names removed. | **58%** balanced accuracy on Indian channels (AUC 0.61). 64% when the model is most confident. |
| Transcript model | Scores the spoken script against the same label | Same method, on 1,499 English transcripts | **49%**, which is chance. It's kept in the file but **given zero weight**. Retrain it on your own channel's data. |
| Craft rules | Hook, pacing, clarity, payoff, ending | Rules from editing practice: greeting openings, filler phrases, repetition, long sentences, beats with no new question or number, late payoff, generic CTA | Not tested against view data |
| Retention estimate | Curve over the script's beats | Simulated: each beat loses viewers based on its risks; a weak hook loses many more early in a short | An estimate, not real audience data |

**Score** = 75% craft + 25% packaging percentile. It is a craft score, not a view forecast.

## Why the accuracy is not 85%

On the same data, **channel subscriber count alone predicts "viral" at 93%**. Script and title text alone gets 52–64%. A tool claiming 85%+ from a script is almost certainly measuring channel size. The honest ceiling with public data is roughly where this model sits.

The path to a much better model is **your own channel's data**: YouTube Studio exports (average view duration, retention per video) paired with your scripts. Put them in the same shape as `data/youtubers.csv` and rerun:

```bash
.venv/bin/python -m scriptdoctor.train_model
```

## Files

- `scriptdoctor/features.py`: feature extraction shared by training and inference
- `scriptdoctor/train_model.py`: trains, cross-validates and saves `models/scriptdoctor.joblib` and `models/model_card.json`
- `scriptdoctor/analyze.py`: the model: packaging score, craft rules, retention curve, fixes
- `scriptdoctor/__main__.py`: command-line tool and local API
- `train.py`, `train_india.py`: the original experiments (data loading and labels)
- `data/`: Kaggle datasets: YouTubers Saying Things (CC0), Mr-beast Transcribed Videos (CC0), Trending YouTube Video Statistics INvideos.csv (CC0)
- `scriptdoctor/static/index.html`: the local web page, powered by the model
- `index.html`: the Claude-powered version of the page (published as a claude.ai artifact)

## Data sources

- https://www.kaggle.com/datasets/praneshmukhopadhyay/youtubers-saying-things
- https://www.kaggle.com/datasets/thedevastator/mrbeast-transcribed-youtube-videos
- https://www.kaggle.com/datasets/datasnaek/youtube-new (India file, Nov 2017 to Jun 2018)
