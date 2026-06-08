# Application Tracker — Semantic Job-Skill Matcher

A Streamlit app that reads a job description and ranks your strongest **selling points** against it using multilingual sentence embeddings, then helps you track applications and interviews in one place.

Built around a single idea: instead of guessing which skills to highlight for a given posting, let a semantic model score the fit between the job description and a curated catalogue of your competencies.

> Originally developed in the context of the Huawei ICT Competition 2025–2026 (multilingual NLP workflow).

---

## Features

- **Semantic job analysis** — paste a job description, get the top-5 most relevant skills with a similarity score (0–100%).
- **Multilingual matching** — works across French and English job descriptions thanks to a multilingual sentence-transformer model (relevant when you work in one language and apply in another).
- **Application tracker** — log applications (company, role, matched skills, status, notes) to a CSV, update their status (`Applied` / `Interview` / `Rejected` / `Offer`), and clean stale records past a configurable threshold.
- **Interview scheduler** — add and view upcoming interviews from the sidebar.

---

## How it works

The matching pipeline is deliberately simple and transparent:

1. Each selling point is turned into a single text blob (`title` + `keywords` + `pitch_2lines`).
2. The job description and every selling point are encoded with `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, with **L2-normalised** embeddings.
3. Because embeddings are normalised, the **cosine similarity** reduces to a dot product (`sp_emb @ jd_emb`).
4. The top-*k* selling points by score are returned, with the score rescaled to a 0–100% range for readability.

This makes the ranking interpretable (it's a single cosine similarity, not a black-box classifier) and language-agnostic within the model's supported languages.

---

## Tech stack

| Layer        | Tools                                                                 |
|--------------|-----------------------------------------------------------------------|
| UI           | Streamlit                                                             |
| NLP / embeddings | sentence-transformers (`paraphrase-multilingual-mpnet-base-v2`)   |
| Data         | pandas, NumPy                                                         |
| Storage      | flat CSV files (no database)                                         |

---

## Project structure

```
.
├── app_Huawei_ICT.py      # main Streamlit application
├── selling_points.csv     # your skills catalogue (input, must be provided)
├── applications.csv       # generated: application log
├── interviews.csv         # generated: scheduled interviews
└── README.md
```

`applications.csv` and `interviews.csv` are created automatically on first use.

---

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install streamlit pandas numpy sentence-transformers
```

On first run, `sentence-transformers` downloads the embedding model (~1 GB), which can take a moment.

---

## Usage

```bash
streamlit run app_Huawei_ICT.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

**Analyse job** tab: enter the company, the role, paste the job description, and click *Analyse*. The app returns the best-matching skills with their scores. Click *Save application* to log it.

**Applications** tab: review your log, update statuses and notes, and optionally clean records older than a chosen number of days (preview mode is on by default so nothing is deleted unintentionally).

**Sidebar**: add upcoming interviews and see them sorted by date.

---

## The selling points catalogue (`selling_points.csv`)

The catalogue is a `;`-separated CSV. Each row is one skill the model can surface. Expected columns:

| Column          | Purpose                                                      |
|-----------------|--------------------------------------------------------------|
| `sp_id`         | unique identifier (e.g. `SP001`)                             |
| `title`         | short skill name (used for display and matching)             |
| `keywords`      | `;`-separated keywords, bilingual — drives most of the match |
| `strength_tags` | high-level tags (`quant`, `finance`, `risk`, …)              |
| `tools_stack`   | associated tools (`Python`, `C++`, `QuantLib`, …)            |
| `pitch_2lines`  | a two-line pitch shown in results and used in matching       |
| `evidence`, `proof_link`, `metrics` | optional supporting detail               |
| `priority`      | manual importance ranking                                    |
| `last_updated`  | date of last edit                                            |

Edit this file to make the tool your own. Richer, more specific `keywords` and `pitch_2lines` generally produce sharper matches.

---

## Notes & known limitations

- **`selling_points.csv` must exist next to the app** and be `;`-separated. If you keep a differently named file, either rename it or update the `SP_CSV` constant in the code.
- The app currently imports `yake` but does not use it. Either remove it from the dependencies or wire in keyword extraction — it is not required to run the app as-is.
- Storage is plain CSV with last-write-wins behaviour; it is not designed for concurrent multi-user editing.
- Scores are relative similarity measures, not calibrated probabilities of fit — use them as a ranking signal, not an absolute judgement.

---

## License
Feel free to use.
