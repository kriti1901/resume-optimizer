"""End-to-end analysis: resume text + JD text -> score, gaps, ATS report."""

from pydantic import BaseModel

from core import extractor
from core.ats import check_ats
from core.gaps import find_gaps
from core.matcher import match_skills, semantic_similarity
from core.schemas import ATSReport, Gap, JDProfile, ResumeProfile, ScoreBreakdown, SkillMatch
from core.scorer import score


class Analysis(BaseModel):
    resume: ResumeProfile
    jd: JDProfile
    required_matches: list[SkillMatch]
    preferred_matches: list[SkillMatch]
    gaps: list[Gap]
    ats: ATSReport
    score: ScoreBreakdown


def analyze_profiles(resume_text: str, jd_text: str, resume: ResumeProfile, jd: JDProfile) -> Analysis:
    """Deterministic part of the pipeline (no LLM calls), so it can be unit-tested."""
    required = match_skills(jd.required_skills, resume.skills, resume_text, resume.experience_bullets)
    preferred = match_skills(jd.preferred_skills, resume.skills, resume_text, resume.experience_bullets)
    ats = check_ats(resume_text, resume.experience_bullets, jd)
    return Analysis(
        resume=resume,
        jd=jd,
        required_matches=required,
        preferred_matches=preferred,
        gaps=find_gaps(required, preferred, resume.skills),
        ats=ats,
        score=score(required, preferred, semantic_similarity(resume_text, jd_text), resume, jd, ats.score),
    )


def analyze(resume_text: str, jd_text: str) -> Analysis:
    return analyze_profiles(resume_text, jd_text, extractor.extract_resume(resume_text), extractor.extract_jd(jd_text))
