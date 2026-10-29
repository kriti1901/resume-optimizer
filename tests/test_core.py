from core import matcher
from core.ats import check_ats
from core.parser import parse_resume, strip_pii
from core.pipeline import analyze_profiles


def test_parse_txt_and_strip_pii(resume_text):
    text = parse_resume("cv.txt", resume_text.encode())
    clean = strip_pii(text)
    assert "priya.sharma@example.com" not in clean
    assert "98765" not in clean
    assert "linkedin.com" not in clean
    assert "FastAPI" in clean


def test_canonical_and_synonyms():
    assert matcher.canonical("JS") == "javascript"
    assert matcher.canonical("k8s") == "kubernetes"
    assert matcher.find_match("AWS", ["Amazon Web Services"]) == "Amazon Web Services"
    assert matcher.find_match("REST API", ["REST APIs"]) == "REST APIs"


def test_no_false_match_java_javascript():
    assert matcher.find_match("Java", ["JavaScript"]) is None
    assert not matcher.mentioned_in("Java", "Expert in JavaScript")


def test_related_skills():
    assert matcher.related_skills("AWS", ["GCP", "Python"]) == ["GCP"]


def test_ats(resume_text, resume_profile, jd_profile):
    report = check_ats(resume_text, resume_profile.experience_bullets, jd_profile)
    assert "Kubernetes" in report.missing_keywords
    assert "Python" in report.found_keywords
    assert report.format_checks["Has an Education section"]
    assert report.format_checks["At least 30% of bullets have numbers"]


def test_full_analysis(resume_text, jd_text, resume_profile, jd_profile):
    a = analyze_profiles(resume_text, jd_text, resume_profile, jd_profile)
    by_skill = {g.skill: g for g in a.gaps}
    assert by_skill["Kubernetes"].category == "transferable"  # has Docker
    assert by_skill["AWS"].category == "transferable"  # has GCP
    assert by_skill["Kafka"].category == "missing"
    assert by_skill["Kafka"].priority == "preferred"
    assert "FastAPI" not in by_skill  # listed and used in a bullet
    assert by_skill["Python"].category == "weak"  # listed, but no bullet mentions it
    assert by_skill["Machine Learning"].category == "weak"  # listed, never used in a bullet
    assert 40 <= a.score.total <= 85
    assert a.score.experience_fit == 0.75


def test_score_is_deterministic(resume_text, jd_text, resume_profile, jd_profile):
    first = analyze_profiles(resume_text, jd_text, resume_profile, jd_profile).score
    second = analyze_profiles(resume_text, jd_text, resume_profile, jd_profile).score
    assert first == second


def test_better_resume_scores_higher(resume_text, jd_text, resume_profile, jd_profile):
    base = analyze_profiles(resume_text, jd_text, resume_profile, jd_profile).score.total
    improved = resume_profile.model_copy(deep=True)
    improved.skills += ["Kubernetes", "AWS", "Kafka"]
    improved.experience_bullets.append("Ran 12 microservices on Kubernetes in AWS with Kafka event streams")
    text = resume_text + "\n- Ran 12 microservices on Kubernetes in AWS with Kafka event streams"
    assert analyze_profiles(text, jd_text, improved, jd_profile).score.total > base
