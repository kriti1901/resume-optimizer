"""ATS keyword coverage and format checks (rule-based, no LLM)."""

import re

from core.matcher import mentioned_in
from core.schemas import ATSReport, JDProfile

SECTION_PATTERNS = {
    "experience": r"\b(experience|employment|work history|projects?)\b",
    "education": r"\beducation\b",
    "skills": r"\b(skills|technologies|tech stack)\b",
}


def check_ats(resume_text: str, bullets: list[str], jd: JDProfile) -> ATSReport:
    keywords = list(dict.fromkeys(jd.required_skills + jd.preferred_skills + jd.keywords))
    found = [k for k in keywords if mentioned_in(k, resume_text)]
    missing = [k for k in keywords if k not in found]
    coverage = len(found) / len(keywords) if keywords else 1.0

    words = len(resume_text.split())
    lower = resume_text.lower()
    quantified = sum(bool(re.search(r"\d", b)) for b in bullets)
    checks = {
        "Text is machine-readable": words >= 100,
        "Length is 300-1200 words": 300 <= words <= 1200,
        "Has an Experience section": bool(re.search(SECTION_PATTERNS["experience"], lower)),
        "Has an Education section": bool(re.search(SECTION_PATTERNS["education"], lower)),
        "Has a Skills section": bool(re.search(SECTION_PATTERNS["skills"], lower)),
        "Contact email present": "@" in resume_text,
        "At least 30% of bullets have numbers": bool(bullets) and quantified / len(bullets) >= 0.3,
    }
    messages = {
        "Text is machine-readable": "Very little text could be extracted. Scanned or image-based PDFs are unreadable to most ATS.",
        "Length is 300-1200 words": f"Your resume has {words} words. Aim for 300-1200 (1-2 pages).",
        "Has an Experience section": "Add a clearly titled 'Experience' or 'Projects' section.",
        "Has an Education section": "Add a clearly titled 'Education' section.",
        "Has a Skills section": "Add a 'Skills' section. ATS parsers look for it.",
        "Contact email present": "Add an email address so the ATS can capture your contact details.",
        "At least 30% of bullets have numbers": "Quantify impact (%, $, time saved, users) in more bullets.",
    }
    failed = [messages[k] for k, ok in checks.items() if not ok]
    if missing:
        failed.insert(0, f"Add these JD keywords where they're true for you: {', '.join(missing[:10])}.")
    fmt_score = sum(checks.values()) / len(checks)
    return ATSReport(
        keyword_coverage=coverage,
        found_keywords=found,
        missing_keywords=missing,
        format_checks=checks,
        format_messages=failed,
        score=0.6 * coverage + 0.4 * fmt_score,
    )
