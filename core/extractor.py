"""LangChain extraction chains: raw resume / JD text -> structured profiles."""

from core.llm import run, structured_chain
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
    chain = structured_chain(RESUME_SYSTEM, "<resume>\n{resume}\n</resume>", ResumeProfile)
    return run(chain, {"resume": strip_pii(resume_text)}, ResumeProfile)


def extract_jd(jd_text: str) -> JDProfile:
    chain = structured_chain(JD_SYSTEM, "<job_description>\n{jd}\n</job_description>", JDProfile)
    return run(chain, {"jd": jd_text}, JDProfile)
