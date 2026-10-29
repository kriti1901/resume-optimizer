"""LangChain rewrite chain: bullet rewrites grounded in the resume's own facts."""

from core.llm import run, structured_chain
from core.parser import strip_pii
from core.schemas import Gap, JDProfile, ResumeProfile, RewriteResult

SYSTEM = """You are an expert resume coach. Rewrite resume bullets so they match a target job description better.

Rules:
- Use only facts already in the original bullet or elsewhere in the resume. Never invent employers, \
numbers, tools or achievements. If a metric would help but none exists, write [add metric] as a placeholder.
- Naturally work in JD keywords the candidate actually has (listed skills or transferable ones).
- Start each bullet with a strong action verb; keep it to one or two lines.
- Pick the 4-6 bullets where a rewrite helps most. Skip bullets that are already strong."""

HUMAN = """<job_description_summary>
Title: {title}
Required: {required}
Preferred: {preferred}
Keywords: {keywords}
</job_description_summary>

<skills_to_surface>{surface}</skills_to_surface>

<resume_bullets>
{bullets}
</resume_bullets>

<full_resume>
{resume}
</full_resume>

Rewrite the bullets per the rules. In `original`, copy the bullet exactly as it appears above."""


def suggest_rewrites(resume_text: str, resume: ResumeProfile, jd: JDProfile, gaps: list[Gap]) -> RewriteResult:
    surface = [g.skill for g in gaps if g.category in ("weak", "transferable")]
    inputs = {
        "title": jd.title,
        "required": ", ".join(jd.required_skills),
        "preferred": ", ".join(jd.preferred_skills),
        "keywords": ", ".join(jd.keywords),
        "surface": ", ".join(surface) or "none",
        "bullets": "\n".join(f"- {b}" for b in resume.experience_bullets),
        "resume": strip_pii(resume_text),
    }
    return run(structured_chain(SYSTEM, HUMAN, RewriteResult), inputs, RewriteResult)
