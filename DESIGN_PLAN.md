# AI Resume Optimizer: Design Plan

> **Update:** the app now uses Google Gemini (`gemini-3.1-pro-preview`) instead of Claude. See the README for setup.

## Context
A portfolio project in Python that uses a hosted LLM API. The user uploads a resume (PDF or DOCX) and pastes a job description (JD). The app returns:
1. A **match score** (0–100) with a breakdown
2. **Skill gaps**: skills the JD requires that are missing or weak in the resume
3. An **ATS keyword check**: keyword coverage and formatting problems
4. **Rewrite suggestions**: resume bullets tailored to the JD

The design aims for low cost, a quick build and a good demo. The score is explainable: it is a hybrid of deterministic NLP and an LLM, not a number the LLM makes up.

---

## Architecture

```
[Streamlit UI] ──► [FastAPI backend] ──► Pipeline:
                                         1. Parse     (resume PDF/DOCX → text)
                                         2. Extract   (LLM → structured JSON: skills, experience, education)
                                         3. Match     (embeddings + keyword overlap)
                                         4. Score     (weighted formula)
                                         5. Gaps      (JD skills − resume skills, by priority)
                                         6. ATS check (keywords + format rules)
                                         7. Rewrite   (LLM, grounded in the gaps)
                                    ──► [SQLite: saved analyses]  (optional)
```

### Scoring formula (explainable)
| Component | Weight | How it's computed |
|---|---|---|
| Required-skill coverage | 40% | Required JD skills matched in the resume (exact, synonym or embedding similarity > 0.8) |
| Semantic similarity | 20% | Cosine similarity of resume and JD embeddings |
| Experience fit | 15% | Years and seniority compared with the JD requirement (LLM-extracted) |
| Preferred-skill coverage | 10% | Nice-to-have skills matched |
| ATS keyword/format | 15% | Keyword density plus a format rule checklist |

Skill gaps fall into three groups: **Missing** (not present), **Weak** (mentioned but not shown in any experience bullet) and **Transferable** (a related skill is present).

---

## Tool options (★ = recommended for this project)

**LLM (hosted API)**
- ★ **Claude**: `claude-sonnet-5-5` for extraction and rewrites, `claude-haiku-5-5` as a cheap fast option. Strong at structured JSON output and tool use.
- OpenAI GPT models, an alternative with a similar API
- Google Gemini, which has a generous free tier

**Embeddings (semantic matching)**
- ★ `sentence-transformers` (`all-MiniLM-L6-v2` or `bge-small`): free and runs locally on CPU
- Voyage AI or OpenAI embeddings: hosted, slightly better quality

**Resume parsing**
- ★ `pdfplumber` (PDF) + `python-docx` (DOCX)
- `PyMuPDF` (fast, keeps the layout), `unstructured` (handles many formats, heavier)
- LLM fallback: send the PDF straight to Claude, which accepts PDF input natively, for messy layouts

**NLP / keywords**
- ★ `spaCy` for tokenizing and lemmatizing, `rapidfuzz` for fuzzy matching
- `KeyBERT` for extracting JD keywords
- A skills taxonomy for normalizing skill names (e.g. "JS" → "JavaScript"): the ESCO or O*NET open datasets, or a small hand-made synonym JSON

**Structured output / orchestration**
- ★ `pydantic` schemas + Anthropic SDK tool use / structured outputs (no framework needed)
- LangChain or LlamaIndex: optional and adds complexity, so skip it for the MVP

**Backend / UI**
- ★ **Streamlit**: the fastest route to a demo, and you could drop FastAPI entirely for the MVP
- FastAPI + a React/Next.js frontend: more polished and shows full-stack skills
- Gradio: an alternative to Streamlit, easy to host on Hugging Face Spaces

**Storage**
- ★ SQLite (via SQLModel)
- Postgres + `pgvector` if you later add search across many JDs

**Vector store** (only if you add many JDs or resumes)
- ChromaDB, FAISS, `pgvector`

**Deployment (free or cheap)**
- ★ Streamlit Community Cloud or Hugging Face Spaces
- Render, Railway or Fly.io for FastAPI

**Testing / eval**
- `pytest`, plus a small golden set of 10–20 resume/JD pairs with hand-labeled scores to check that scores are consistent

---

## Recommended MVP stack
Python 3.12 · Streamlit · Anthropic SDK (Claude Sonnet/Haiku) · pydantic · pdfplumber + python-docx · sentence-transformers · spaCy + rapidfuzz · SQLite · deployed on Streamlit Cloud.

## Where the project will be saved
- Nothing has been built yet. So far only this plan exists.
- Once approved, the code goes in `/home/user/resume-optimizer/`, inside the working directory, so the Claude app can open it.
- This cloud container is temporary. To keep the code, either push it to a GitHub repo or have the files sent to you for download.

## Proposed project structure
```
resume-optimizer/
  app.py                 # Streamlit UI
  core/
    parser.py            # PDF/DOCX → text
    extractor.py         # LLM → pydantic ResumeProfile / JDProfile
    matcher.py           # embeddings + fuzzy skill matching
    scorer.py            # weighted score + breakdown
    gaps.py              # missing/weak/transferable
    ats.py               # keyword + format checks
    rewriter.py          # LLM bullet rewrites
    schemas.py           # pydantic models
  data/skills_synonyms.json
  tests/ (golden pairs)
  .env                   # ANTHROPIC_API_KEY
```

## Build phases
1. **Week 1**: parsing, LLM extraction into pydantic schemas, a Streamlit upload screen
2. **Week 2**: matching, the scoring formula, the skill-gap view (charts with Plotly)
3. **Week 3**: ATS checker and rewrite suggestions shown as a before/after diff
4. **Week 4**: golden-set evaluation, prompt caching to cut cost, deployment, README with screenshots

## Key design notes
- **Determinism**: set temperature 0 for extraction and cache results by a hash of resume + JD, so the same inputs always give the same score
- **Grounding**: rewrites must not invent experience. The prompt allows rephrasing only of facts already in the resume.
- **Privacy**: don't keep resumes by default, and strip personal info (email, phone) before LLM calls
- **Cost**: Haiku for extraction and Sonnet for rewrites cost roughly a cent or two per analysis

## Verification
- Unit tests for the parser, matcher and scorer using fixture resumes
- Golden set: scores land within ±10 of the hand labels, and the same input gives the same score
- Manual end-to-end check: upload a sample resume and JD in Streamlit, then confirm the score breakdown, the gap list, the ATS warnings and the rewrites
