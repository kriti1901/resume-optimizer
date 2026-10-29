from pathlib import Path

import pytest

from core.schemas import JDProfile, ResumeProfile

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def resume_text():
    return (FIX / "resume.txt").read_text()


@pytest.fixture
def jd_text():
    return (FIX / "jd.txt").read_text()


# What the LLM extractor is expected to return for the fixtures, so tests run without an API key.
@pytest.fixture
def resume_profile():
    return ResumeProfile(
        skills=["Python", "FastAPI", "Flask", "PostgreSQL", "Docker", "Git", "REST APIs", "GCP", "Pandas", "Machine Learning"],
        experience_bullets=[
            "Built REST APIs in FastAPI serving 2M requests per day with 99.9% uptime",
            "Designed PostgreSQL schemas and cut query latency by 40% through indexing",
            "Containerized 6 services with Docker and deployed them on GCP",
            "Mentored 2 junior developers on code review practices",
            "Wrote Flask services for internal reporting dashboards",
            "Automated data cleaning with Pandas, saving the team 5 hours per week",
        ],
        job_titles=["Software Engineer", "Junior Developer"],
        total_years_experience=3,
        education=["B.Tech in Computer Science, Delhi Technological University, 2021"],
    )


@pytest.fixture
def jd_profile():
    return JDProfile(
        title="Senior Backend Engineer - Payments",
        seniority="senior",
        required_skills=["Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes", "AWS", "REST API"],
        preferred_skills=["Kafka", "Machine Learning", "Terraform"],
        min_years_experience=4,
        keywords=["microservices", "payments", "CI/CD"],
    )
