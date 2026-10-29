"""LLM bullet rewrites, grounded in the resume's own facts."""

from core.llm import parse
from core.parser import strip_pii
from core.schemas import Gap, JDProfile, ResumeProfile, RewriteResult

SYSTEM = """You are an expert resume coach. Rewrite resume bullets so they match a target job description better.

Rules:
- Use only facts already in the original bullet or elsewhere in the resume. Never invent employers, \
numbers, tools or achievements. If a metric would help but none exists, write [add metric] as a placeholder.
- Naturally work in JD keywords the candidate actually has (listed skills or transferable ones).
- Start each bullet with a strong action verb; keep it to one or two lines.
- Pick the 4-6 bullets where a rewrite helps most. Skip bullets that are already strong."""


def suggest_rewrites(resume_text: str, resume: ResumeProfile, jd: JDProfile, gaps: list[Gap]) -> RewriteResult:
    weak = [g.skill for g in gaps if g.category in ("weak", "transferable")]
    bullets = "\n".join(f"- {b}" for b in resume.experience_bullets)
    user = f"""<job_description_summary>
Title: {jd.title}
Required: {', '.join(jd.required_skills)}
Preferred: {', '.join(jd.preferred_skills)}
Keywords: {', '.join(jd.keywords)}
</job_description_summary>

<skills_to_surface>{', '.join(weak) or 'none'}</skills_to_surface>

<resume_bullets>
{bullets}
</resume_bullets>

<full_resume>
{strip_pii(resume_text)}
</full_resume>

Rewrite the bullets per the rules. In `original`, copy the bullet exactly as it appears above."""
    return parse(SYSTEM, user, RewriteResult, max_tokens=32000)
