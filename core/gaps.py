"""Classify JD skills the resume doesn't fully cover."""

from core.matcher import related_skills
from core.schemas import Gap, SkillMatch


def find_gaps(required: list[SkillMatch], preferred: list[SkillMatch], resume_skills: list[str]) -> list[Gap]:
    gaps = []
    for priority, matches in (("required", required), ("preferred", preferred)):
        for m in matches:
            if m.matched_resume_skill is None:
                related = related_skills(m.jd_skill, resume_skills)
                if related:
                    gaps.append(Gap(
                        skill=m.jd_skill, category="transferable", priority=priority,
                        note=f"Not listed, but you have {', '.join(related)}. Mention transferable experience.",
                    ))
                else:
                    gaps.append(Gap(
                        skill=m.jd_skill, category="missing", priority=priority,
                        note="Not found in your resume.",
                    ))
            elif not m.in_experience:
                gaps.append(Gap(
                    skill=m.jd_skill, category="weak", priority=priority,
                    note="Listed as a skill but not shown in any experience bullet. Add a bullet that uses it.",
                ))
    order = {"required": 0, "preferred": 1}
    cat = {"missing": 0, "transferable": 1, "weak": 2}
    return sorted(gaps, key=lambda g: (order[g.priority], cat[g.category]))
