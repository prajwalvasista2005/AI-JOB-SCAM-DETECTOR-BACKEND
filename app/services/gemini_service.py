
import json

from google import genai

from app.core.config import settings


client = genai.Client(api_key=settings.GEMINI_API_KEY)


def analyze_job_content(content: str) -> dict:
    prompt = f"""
You are an assistant that assesses potential job scams.

Analyze the following job posting for warning signs, including:
- Requests for registration fees or upfront payments
- Guaranteed employment claims
- Unusual interview or hiring practices
- Suspicious contact methods
- Unrealistic promises or pressure tactics

Do not assume a job is fraudulent just because it has one warning sign.
A legitimate job can have ordinary hiring language.
Assess the full context and distinguish explicit evidence from uncertainty.

Return ONLY a JSON object with these fields:
- risk_score: integer from 0 to 100
- explanation: short explanation of the assessment
- indicators: list of specific warning signs found

Job posting:
<job_posting>
{content}
</job_posting>
"""

    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
        },
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response")

    result = json.loads(response.text)

    risk_score = result.get("risk_score")
    explanation = result.get("explanation")
    indicators = result.get("indicators")

    if (
        isinstance(risk_score, bool)
        or not isinstance(risk_score, int)
        or not 0 <= risk_score <= 100
    ):
        raise ValueError("Gemini returned an invalid risk score")

    if not isinstance(explanation, str):
        raise ValueError("Gemini returned an invalid explanation")

    if (
        not isinstance(indicators, list)
        or not all(isinstance(item, str) for item in indicators)
    ):
        raise ValueError("Gemini returned invalid indicators")

    return {
        "risk_score": risk_score,
        "explanation": explanation,
        "indicators": indicators,
    }
