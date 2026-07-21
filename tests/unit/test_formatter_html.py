from ter_calculator.formatter_html import format_html
from ter_calculator.models import (
    ClassificationExplanation,
    ClassifiedSpan,
    SpanLabel,
    SpanPhase,
    TERResult,
    TokenSpan,
    UncertaintyReport,
)


def _result() -> TERResult:
    span = TokenSpan(
        text="A concise assistant answer.",
        phase=SpanPhase.GENERATION,
        position=0,
        token_count=5,
        source_message_uuid="assistant-1",
        source_role="assistant",
    )
    classified = ClassifiedSpan(
        span=span,
        label=SpanLabel.ALIGNED_RESPONSE,
        confidence=0.92,
        cosine_similarity=0.81,
        explanation=ClassificationExplanation(
            reason_code="aligned",
            summary="The answer directly addresses the prompt.",
        ),
    )
    return TERResult(
        session_id="html-test",
        aggregate_ter=1.0,
        raw_ratio=1.0,
        phase_scores={"generation": 1.0},
        total_tokens=5,
        aligned_tokens=5,
        waste_tokens=0,
        classified_spans=[classified],
        uncertainty=UncertaintyReport(
            mean_confidence=0.92,
            token_weighted_confidence=0.92,
            low_confidence_tokens=0,
            low_confidence_share=0.0,
            interval_lower=1.0,
            interval_upper=1.0,
            bootstrap_samples=100,
            span_count=1,
            reliability="high",
        ),
    )


def test_format_html_is_standalone_and_interactive():
    rendered = format_html(_result())
    assert rendered.startswith("<!doctype html>")
    assert "Token Efficiency Report" in rendered
    assert 'id="ter-data"' in rendered
    assert "Token composition" in rendered
    assert "Alignment vs confidence" in rendered
    assert "Span inspector" in rendered
    assert "Download analysis JSON" in rendered
    assert "https://" not in rendered


def test_format_html_escapes_script_termination():
    result = _result()
    result.classified_spans[0].span.text = "</script><script>alert(1)</script>"
    rendered = format_html(result)
    assert "<\\/script>" in rendered
    assert "</script><script>alert(1)</script>" not in rendered
