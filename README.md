# Football Intelligence Copilot

An AI-assisted platform for tactical football analysis — turning raw match event data into
tools an analyst or coach could actually use: pass networks, shot maps, an expected goals (xG)
model, and an Expected Threat (xT) possession-value model, served through a FastAPI backend.

Built end-to-end on [StatsBomb's free open data](https://github.com/statsbomb/open-data)
(FA Women's Super League, 2018/19–2023/24), from raw event ingestion through model training,
validation, and API serving.

## Why this project

Football clubs face a familiar problem for any data-rich organisation: plenty of raw data,
not enough tooling that turns it into decisions. This project is a small, honest version of
that pipeline — not a attempt to replicate commercial analytics platforms, but a demonstration
of the full path from raw event data to a validated model to a served, queryable result.

## Architecture

```
StatsBomb Open Data
        │
        ▼
  Ingestion Layer (src/ingestion/)
  - normalizes raw StatsBomb JSON into a canonical event schema
  - caches per-match and combined datasets as Parquet
        │
        ▼
  Analytics Layer (src/analytics/)
  - progressive pass classification
  - xG model (logistic regression: distance + angle → goal probability)
  - xT model (zone-based Markov-chain possession value)
        │
        ▼
  Training Pipelines (scripts/)
  - match-level train/val/test split, offline model fitting
  - saved artifacts: models/xg_model.pkl, models/xt_grid.npy
        │
        ▼
  Visualisation Layer (src/visualisation/)
  - pass networks, shot maps, xT heatmaps (mplsoccer)
        │
        ▼
  API Layer (src/api/)
  - FastAPI service, models loaded once at startup
  - GET /matches/{id}/pass-network, /matches/{id}/shot-map, /xt
```

Analytics functions are pure and reusable; training/data-pipeline code is kept separate from
inference code, so the API loads pre-trained artifacts rather than retraining on every request.

## Features

### Pass Networks

<img width="960" height="690" alt="image" src="https://github.com/user-attachments/assets/97ba74e0-ffff-444e-8899-0362ac4a4bed" />


Node position = average location of each player's passes; edge width = pass frequency between
a given pair of players. Reading this match's network: **Little sits centrally as the team's
structural pivot**, with the highest connectivity on the pitch, while **the right side (Foord,
Catley, Russo) shows denser combination play than the left**, suggesting Arsenal favoured
progression through the right half-space in this fixture.

*Known simplification: the network is computed across the full match rather than cut at the
first substitution, so a substitute's average position reflects only their partial minutes on
the pitch.*

### xG Model

A deliberately minimal 2-feature logistic regression (shot distance + shot angle to goal),
trained on ~11,800 shots across 457 WSL matches, with a **match-level** train/validation/test
split to avoid leaking same-match shot context across splits.

| Split | Log loss | Brier score | ROC AUC |
|---|---|---|---|
| Validation | 0.297 | 0.084 | 0.717 |
| Test | 0.335 | 0.098 | 0.713 |

Fitted coefficients: `distance = -0.062` (farther → lower probability, as expected),
`angle = 0.981` (wider angle → higher probability, as expected). An AUC of ~0.71–0.72 sits
below full commercial xG models (typically ~0.75–0.80), which is expected and explainable:
this model deliberately omits shot technique, defender positions, and goalkeeper positioning —
features a production-grade model would include.

### Shot Maps

<img width="583" height="790" alt="image" src="https://github.com/user-attachments/assets/b06751cb-4ca3-442d-9901-28e475ef8db2" />


Marker size = predicted xG, color = actual outcome. This match: Arsenal scored 2 goals from a
combined 0.74 xG — an overperformance worth noting rather than treating as a modeling error;
it reflects finishing quality above the underlying chance quality created, which is exactly
the kind of distinction an xG baseline is meant to surface.

### Expected Threat (xT)

Implemented from first principles following the framework introduced by
[Karun Singh (2018)](https://karun.in/blog/expected-threat.html), building on Sarah Rudd's
original 2011 concept: the pitch is divided into a 16×12 zone grid, and each zone's value is
solved iteratively as a Markov-chain fixed point — combining the immediate value of shooting
from a zone with the value of the zones a team's completed passes/carries tend to lead to.

<img width="928" height="690" alt="image" src="https://github.com/user-attachments/assets/921f2a13-98af-4909-9b67-2db17e76f601" />


**A modeling decision worth detailing:** the initial implementation only counted shots and
completed passes/carries in its zone statistics, implicitly assuming possession is never lost.
Testing this assumption directly (comparing kept-vs-excluded action rates by pitch zone, then
building a turnover-aware variant with an explicit `turnover_prob` and `turnover_value = 0`)
showed the naive model **overvalued deep buildup zones by roughly 80–90% relative to zones
near goal** — a substantial, quantified bias, corrected in the turnover-aware version.

<img width="928" height="690" alt="image" src="https://github.com/user-attachments/assets/3dc16451-cabd-45d8-9fa7-4eecdf029abc" />
<img width="939" height="690" alt="image" src="https://github.com/user-attachments/assets/3284a951-0cf5-4c7a-85c1-25a5aeb5e55a" />
<img width="925" height="690" alt="image" src="https://github.com/user-attachments/assets/4d425e9d-11f1-4a70-be31-0d7065870ae1" />


*Known limitation, and a real next step: `turnover_value = 0` still treats a lost possession
as equally costless everywhere, when in reality losing the ball in your own third is far more
dangerous (higher opponent counter-attack value) than losing it near the opponent's goal.
A proper fix requires building the opponent's own xT surface and crediting turnovers with a
negative value based on the (mirrored) zone the ball was lost in — scoped but not yet built.*

## Setup

```bash
pyenv virtualenv 3.12.11 football-intelligence
pyenv local football-intelligence
pip install -r requirements.txt
```

## Running the pipelines

```bash
# train the xG model (pulls + caches WSL event data, trains, validates, saves artifact)
python -m scripts.train_xg_model

# build the xT grid (requires the trained xG model)
python -m scripts.build_xt_grid

# run the API
uvicorn src.api.main:app --reload
# interactive docs: http://127.0.0.1:8000/docs
```

## Project structure

```
├── data/               # cached raw + processed event data (gitignored)
├── models/             # trained xG model, xT grid
├── notebooks/          # exploratory analysis
├── scripts/            # one-off training/build pipelines
├── src/
│   ├── ingestion/      # StatsBomb loading + normalization + caching
│   ├── analytics/      # progressive passes, xG, xT
│   ├── visualisation/  # pass networks, shot maps, xT heatmaps
│   └── api/            # FastAPI service
└── tests/
```

## What I'd build next

- Opponent-aware xT: negative turnover value based on the opposing team's own threat surface
- Per-team xT surfaces (e.g. Arsenal's own zone valuations vs. league average), to capture
  tactical identity rather than one blended league-wide model
- A minimal frontend for browsing matches interactively, rather than querying the API directly
- Extending the xG model's feature set (shot technique, assist type) once a larger, richer
  dataset is available

## Data & attribution

Match event data provided free by [StatsBomb](https://statsbomb.com/) via their open data
repository, used under their open data license. Analysis and models are original work built
on top of that data.
