# AI Resume Optimizer

Upload a resume and paste a job description to get:

- **Match score (0–100)** with an explainable breakdown
- **Skill gaps**: missing, transferable (you have a related skill) or weak (listed but never shown in experience)
- **ATS check**: keyword coverage plus format rules
- **Rewrite suggestions**: bullets tailored to the JD, using only facts already in your resume

Built with Python, Streamlit and the Claude API.

## How the score works

Claude only *extracts* structured data (skills, bullets, years of experience). The score itself is computed by deterministic code, so it is explainable and repeatable.

| Component | Weight | How |
|---|---|---|
| Required skills | 40% | Share of must-have JD skills found (exact, synonym or fuzzy match). A skill listed but never used in a bullet counts 0.75 |
| Overall similarity | 20% | TF-IDF cosine similarity of resume and JD |
| Experience fit | 15% | Resume years ÷ JD minimum years (capped at 1) |
| Nice-to-have skills | 10% | Same as required skills, for preferred skills |
| ATS | 15% | 60% keyword coverage + 40% format checks |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then put your Anthropic API key in .env
streamlit run app.py
```

Get an API key at https://console.anthropic.com.

## Models and cost

- Extraction: `claude-haiku-5-5` (cheap and fast)
- Rewrites: `claude-sonnet-5-5`, with server-side refusal fallbacks enabled

Override them with the `EXTRACT_MODEL` / `REWRITE_MODEL` environment variables. A full analysis with rewrites costs about a cent or two. Results are cached per resume + JD, so re-running costs nothing.

## Privacy

Email addresses, phone numbers and profile URLs are stripped before any text is sent to the API. Nothing is stored on disk.

## Project layout

```
app.py              Streamlit UI
core/parser.py      PDF/DOCX/TXT -> text, PII stripping
core/llm.py         Claude structured-output wrapper
core/extractor.py   Resume/JD -> pydantic profiles
core/matcher.py     Synonyms, fuzzy matching, similarity
core/gaps.py        Missing / transferable / weak skills
core/ats.py         Keyword and format checks
core/scorer.py      Weighted score
core/rewriter.py    Bullet rewrites
core/pipeline.py    Ties it together
data/skills_synonyms.json   Skill aliases and related-skill families
tests/              pytest suite (no API key needed)
```

## Tests

```bash
pytest
```

## Deploy (free)

Push to GitHub, then create an app on [Streamlit Community Cloud](https://share.streamlit.io) pointing at `app.py`, and add `ANTHROPIC_API_KEY` under **Secrets**.

## Ideas for next steps

- Add 10–20 hand-scored resume/JD pairs and tune the weights against them
- Swap TF-IDF for `sentence-transformers` embeddings for better similarity
- Export the improved resume as PDF/DOCX
