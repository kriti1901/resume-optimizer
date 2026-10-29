"""Weighted, explainable match score."""

from core.schemas import JDProfile, ResumeProfile, ScoreBreakdown, SkillMatch

WEIGHTS = {
    "required_skills": 0.40,
    "semantic_similarity": 0.20,
    "experience_fit": 0.15,
    "preferred_skills": 0.10,
    "ats": 0.15,
}


def coverage(matches: list[SkillMatch]) -> float:
    """Share of JD skills matched. A skill listed but never shown in experience counts as 0.75."""
    if not matches:
        return 1.0
    total = sum(0 if m.matched_resume_skill is None else (1.0 if m.in_experience else 0.75) for m in matches)
    return total / len(matches)


def experience_fit(resume: ResumeProfile, jd: JDProfile) -> float:
    if jd.min_years_experience <= 0:
        return 1.0
    return min(1.0, resume.total_years_experience / jd.min_years_experience)


def score(required: list[SkillMatch], preferred: list[SkillMatch], similarity: float,
          resume: ResumeProfile, jd: JDProfile, ats_score: float) -> ScoreBreakdown:
    parts = {
        "required_skills": coverage(required),
        "semantic_similarity": similarity,
        "experience_fit": experience_fit(resume, jd),
        "preferred_skills": coverage(preferred),
        "ats": ats_score,
    }
    total = round(100 * sum(WEIGHTS[k] * v for k, v in parts.items()))
    return ScoreBreakdown(**parts, total=total)
