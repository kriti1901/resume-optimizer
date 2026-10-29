"""Skill normalization, fuzzy skill matching and semantic similarity."""

import json
import re
from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.schemas import SkillMatch

DATA = Path(__file__).resolve().parent.parent / "data" / "skills_synonyms.json"
FUZZY_THRESHOLD = 90


@lru_cache
def _taxonomy() -> tuple[dict[str, str], dict[str, list[str]], list[set[str]]]:
    raw = json.loads(DATA.read_text())
    alias_to_canon: dict[str, str] = {}
    canon_aliases: dict[str, list[str]] = {}
    for canon, aliases in raw["synonyms"].items():
        alias_to_canon[canon] = canon
        canon_aliases[canon] = [canon, *aliases]
        for a in aliases:
            alias_to_canon[a] = canon
    related = [{canonical(s, alias_to_canon) for s in group} for group in raw["related"]]
    return alias_to_canon, canon_aliases, related


def canonical(skill: str, alias_to_canon: dict[str, str] | None = None) -> str:
    if alias_to_canon is None:
        alias_to_canon = _taxonomy()[0]
    s = re.sub(r"\s+", " ", skill.strip().lower())
    return alias_to_canon.get(s, s)


def aliases(skill: str) -> list[str]:
    c = canonical(skill)
    return _taxonomy()[1].get(c, [c])


def mentioned_in(skill: str, text: str) -> bool:
    """True if the skill (or one of its aliases) appears as a whole word in text."""
    lower = text.lower()
    for a in aliases(skill):
        if re.search(rf"(?<![\w+#]){re.escape(a)}(?![\w+#])", lower):
            return True
    return False


def find_match(jd_skill: str, resume_skills: list[str]) -> str | None:
    target = canonical(jd_skill)
    best, best_score = None, 0.0
    for rs in resume_skills:
        c = canonical(rs)
        if c == target:
            return rs
        score = fuzz.token_sort_ratio(c, target)
        if score > best_score:
            best, best_score = rs, score
    return best if best_score >= FUZZY_THRESHOLD else None


def match_skills(jd_skills: list[str], resume_skills: list[str], resume_text: str, bullets: list[str]) -> list[SkillMatch]:
    bullet_text = "\n".join(bullets)
    results = []
    for js in jd_skills:
        m = find_match(js, resume_skills)
        # Fall back to a raw text search: the extractor may have missed a skill the resume does mention.
        if m is None and mentioned_in(js, resume_text):
            m = js
        results.append(SkillMatch(
            jd_skill=js,
            matched_resume_skill=m,
            in_experience=m is not None and (mentioned_in(js, bullet_text) or mentioned_in(m, bullet_text)),
        ))
    return results


def related_skills(skill: str, resume_skills: list[str]) -> list[str]:
    """Resume skills in the same family as `skill` (e.g. GCP for an AWS requirement)."""
    target = canonical(skill)
    resume_canon = {canonical(s): s for s in resume_skills}
    out = []
    for group in _taxonomy()[2]:
        if target in group:
            out += [orig for c, orig in resume_canon.items() if c in group and c != target]
    return sorted(set(out))


def semantic_similarity(resume_text: str, jd_text: str) -> float:
    """Cosine similarity of TF-IDF vectors (0..1). Rescaled because raw TF-IDF scores for
    a good match usually land around 0.3-0.5."""
    if not resume_text.strip() or not jd_text.strip():
        return 0.0
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    m = vec.fit_transform([resume_text, jd_text])
    raw = float(cosine_similarity(m[0], m[1])[0, 0])
    return min(1.0, raw / 0.5)
