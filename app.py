"""Streamlit UI for the AI Resume Optimizer. Run with: streamlit run app.py"""

import os

import plotly.graph_objects as go
import streamlit as st

from core import extractor, rewriter
from core.llm import LLMError
from core.parser import parse_resume
from core.pipeline import analyze_profiles
from core.scorer import WEIGHTS

st.set_page_config(page_title="AI Resume Optimizer", page_icon="📄", layout="wide")


# Cached by input text, so the same resume + JD always gives the same score and costs nothing to re-run.
@st.cache_data(show_spinner=False, max_entries=50)
def cached_resume_profile(text: str):
    return extractor.extract_resume(text)


@st.cache_data(show_spinner=False, max_entries=50)
def cached_jd_profile(text: str):
    return extractor.extract_jd(text)


@st.cache_data(show_spinner=False, max_entries=50)
def cached_rewrites(resume_text: str, _analysis, jd_key: str):
    return rewriter.suggest_rewrites(resume_text, _analysis.resume, _analysis.jd, _analysis.gaps)


LABELS = {
    "required_skills": "Required skills",
    "semantic_similarity": "Overall similarity",
    "experience_fit": "Experience fit",
    "preferred_skills": "Nice-to-have skills",
    "ats": "ATS keywords & format",
}

st.title("📄 AI Resume Optimizer")
st.caption("Upload your resume and paste a job description to get a match score, skill gaps, an ATS check and rewrite suggestions.")

if not os.getenv("ANTHROPIC_API_KEY"):
    st.warning("Set ANTHROPIC_API_KEY in a `.env` file or your environment before running an analysis.")

col1, col2 = st.columns(2)
with col1:
    upload = st.file_uploader("Resume (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
with col2:
    jd_text = st.text_area("Job description", height=220, placeholder="Paste the full job description here...")

if st.button("Analyze", type="primary", disabled=not (upload and jd_text.strip())):
    try:
        resume_text = parse_resume(upload.name, upload.getvalue())
    except Exception as e:  # pdfplumber/python-docx raise many different errors on bad files
        st.error(f"Couldn't read that file: {e}")
        st.stop()
    if len(resume_text.split()) < 30:
        st.error("Almost no text could be extracted. Is this a scanned or image-only PDF?")
        st.stop()
    try:
        with st.spinner("Reading your resume and the job description..."):
            resume = cached_resume_profile(resume_text)
            jd = cached_jd_profile(jd_text.strip())
    except LLMError as e:
        st.error(str(e))
        st.stop()
    st.session_state.resume_text = resume_text
    st.session_state.analysis = analyze_profiles(resume_text, jd_text, resume, jd)
    st.session_state.pop("rewrites", None)

analysis = st.session_state.get("analysis")
if analysis is None:
    st.stop()

s = analysis.score
st.divider()
st.subheader(f"Results for: {analysis.jd.title}")
left, right = st.columns([1, 2])
with left:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=s.total, number={"suffix": " / 100"},
        title={"text": "Match score"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2563eb"},
               "steps": [{"range": [0, 50], "color": "#fee2e2"},
                         {"range": [50, 75], "color": "#fef3c7"},
                         {"range": [75, 100], "color": "#dcfce7"}]},
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=10, l=40, r=40))
    st.plotly_chart(fig, use_container_width=True)
with right:
    keys = list(WEIGHTS)
    fig = go.Figure(go.Bar(
        x=[round(getattr(s, k) * 100) for k in keys],
        y=[f"{LABELS[k]} ({int(WEIGHTS[k] * 100)}%)" for k in keys],
        orientation="h", marker_color="#2563eb", text=[f"{round(getattr(s, k) * 100)}%" for k in keys],
    ))
    fig.update_layout(height=280, xaxis=dict(range=[0, 100], title="Component score"),
                      yaxis=dict(autorange="reversed"), margin=dict(t=30, b=10, l=10, r=10),
                      title="Score breakdown (weight in brackets)")
    st.plotly_chart(fig, use_container_width=True)

tab_gaps, tab_ats, tab_rewrite, tab_raw = st.tabs(["Skill gaps", "ATS check", "Rewrite suggestions", "Extracted data"])

with tab_gaps:
    icons = {"missing": "🔴 Missing", "transferable": "🟡 Transferable", "weak": "🟠 Weak"}
    if not analysis.gaps:
        st.success("No skill gaps found. Your resume covers every listed skill.")
    for priority in ("required", "preferred"):
        rows = [g for g in analysis.gaps if g.priority == priority]
        if rows:
            st.subheader("Required skills" if priority == "required" else "Nice-to-have skills")
            st.table([{"Skill": g.skill, "Status": icons[g.category], "What to do": g.note} for g in rows])
    matched = [m.jd_skill for m in analysis.required_matches + analysis.preferred_matches if m.matched_resume_skill and m.in_experience]
    if matched:
        st.markdown("**✅ Fully covered:** " + ", ".join(matched))

with tab_ats:
    a = analysis.ats
    c1, c2 = st.columns(2)
    c1.metric("Keyword coverage", f"{round(a.keyword_coverage * 100)}%")
    c2.metric("Format checks passed", f"{sum(a.format_checks.values())} / {len(a.format_checks)}")
    for name, ok in a.format_checks.items():
        st.markdown(f"{'✅' if ok else '❌'} {name}")
    if a.format_messages:
        st.subheader("Fixes")
        for msg in a.format_messages:
            st.markdown(f"- {msg}")
    if a.missing_keywords:
        st.markdown("**Missing keywords:** " + ", ".join(f"`{k}`" for k in a.missing_keywords))

with tab_rewrite:
    st.caption("Rewrites only rephrase facts already in your resume. Replace any [add metric] placeholder with a real number.")
    if st.button("Generate rewrite suggestions"):
        try:
            with st.spinner("Rewriting bullets for this job..."):
                st.session_state.rewrites = cached_rewrites(
                    st.session_state.resume_text, analysis, analysis.jd.model_dump_json()
                )
        except LLMError as e:
            st.error(str(e))
    rw = st.session_state.get("rewrites")
    if rw:
        st.info(f"**Summary tip:** {rw.summary_tip}")
        for sug in rw.suggestions:
            with st.container(border=True):
                b, a_ = st.columns(2)
                b.markdown("**Before**")
                b.write(sug.original)
                a_.markdown("**After**")
                a_.write(sug.rewritten)
                st.caption(f"{sug.reason}" + (f" · Keywords: {', '.join(sug.keywords_added)}" if sug.keywords_added else ""))

with tab_raw:
    c1, c2 = st.columns(2)
    c1.markdown("**Resume**")
    c1.json(analysis.resume.model_dump())
    c2.markdown("**Job description**")
    c2.json(analysis.jd.model_dump())
