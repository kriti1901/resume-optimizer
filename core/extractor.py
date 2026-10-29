"""LLM extraction: raw resume / JD text -> structured profiles."""

from core.llm import EXTRACT_MODEL, parse
from core.parser import strip_pii
from core.schemas import JDProfile, ResumeProfile

RESUME_SYSTEM = (
    "You extract structured data from resumes. Copy skills and bullets as written; "
    "do not infer skills that are not mentioned. Use short canonical skill names "
    "(e.g. 'Python', 'AWS', 'React'). Split combined entries like 'Python/SQL' into separate skills."
)

JD_SYSTEM = (
    "You extract hiring requirements from job descriptions. Put a skill in required_skills only if "
    "the JD presents it as a must-have; 'nice to have', 'bonus', 'plus' or 'preferred' items go in "
    "preferred_skills. Use short canonical skill names. Do not invent requirements."
)


def extract_resume(resume_text: str) -> ResumeProfile:
    return parse(EXTRACT_MODEL, RESUME_SYSTEM, f"<resume>\n{strip_pii(resume_text)}\n</resume>", ResumeProfile)


def extract_jd(jd_text: str) -> JDProfile:
    return parse(EXTRACT_MODEL, JD_SYSTEM, f"<job_description>\n{jd_text}\n</job_description>", JDProfile)
