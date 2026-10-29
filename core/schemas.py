"""Pydantic models shared across the pipeline."""

from pydantic import BaseModel, Field


# ---------- LLM extraction outputs ----------

class ResumeProfile(BaseModel):
    skills: list[str] = Field(description="Every technical and soft skill, tool, language or framework mentioned anywhere")
    experience_bullets: list[str] = Field(description="Each experience/project bullet point, verbatim")
    job_titles: list[str]
    total_years_experience: float = Field(description="Total professional years; 0 if a student/no experience")
    education: list[str]


class JDProfile(BaseModel):
    title: str
    seniority: str = Field(description="One of: intern, junior, mid, senior, lead, unknown")
    required_skills: list[str] = Field(description="Must-have skills, tools and technologies")
    preferred_skills: list[str] = Field(description="Nice-to-have / bonus skills")
    min_years_experience: float = Field(description="Minimum years required; 0 if not stated")
    keywords: list[str] = Field(description="Other important ATS keywords: domains, methodologies, certifications")


class RewriteSuggestion(BaseModel):
    original: str
    rewritten: str
    reason: str
    keywords_added: list[str]


class RewriteResult(BaseModel):
    suggestions: list[RewriteSuggestion]
    summary_tip: str = Field(description="One-sentence suggestion for the resume summary/headline")


# ---------- Pipeline outputs ----------

class SkillMatch(BaseModel):
    jd_skill: str
    matched_resume_skill: str | None
    in_experience: bool  # skill is evidenced in an experience bullet


class Gap(BaseModel):
    skill: str
    category: str  # "missing" | "weak" | "transferable"
    priority: str  # "required" | "preferred"
    note: str


class ATSReport(BaseModel):
    keyword_coverage: float  # 0..1
    found_keywords: list[str]
    missing_keywords: list[str]
    format_checks: dict[str, bool]
    format_messages: list[str]
    score: float  # 0..1


class ScoreBreakdown(BaseModel):
    required_skills: float
    semantic_similarity: float
    experience_fit: float
    preferred_skills: float
    ats: float
    total: int  # 0..100
