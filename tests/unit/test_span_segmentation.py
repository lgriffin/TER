from ter_calculator.loader import segment_spans
from ter_calculator.models import ContentBlock, Message, Session, SpanPhase
from ter_calculator.span_segmentation import SegmentationConfig, segment_text


def _long_sentence(label: str, repetitions: int = 18) -> str:
    return f"{label} " + "useful implementation detail " * repetitions + "."


def test_disabled_segmentation_preserves_parent_block():
    text = _long_sentence("First") + "\n\n" + _long_sentence("Second")
    segments = segment_text(text, SegmentationConfig(enabled=False, max_tokens=20))
    assert len(segments) == 1
    assert segments[0].text == text
    assert segments[0].char_start == 0
    assert segments[0].char_end == len(text)


def test_enabled_segmentation_splits_long_paragraphs_and_preserves_offsets():
    text = _long_sentence("First") + "\n\n" + _long_sentence("Second")
    segments = segment_text(
        text,
        SegmentationConfig(enabled=True, min_tokens=4, max_tokens=35),
    )
    assert len(segments) >= 2
    for segment in segments:
        assert text[segment.char_start : segment.char_end] == segment.text


def test_small_segments_are_merged():
    text = "# Plan\n\n" + _long_sentence("Implement", 8)
    segments = segment_text(
        text,
        SegmentationConfig(enabled=True, min_tokens=8, max_tokens=100),
    )
    assert len(segments) == 1
    assert segments[0].text.startswith("# Plan")


def test_invalid_configuration_is_rejected():
    try:
        SegmentationConfig(enabled=True, min_tokens=20, max_tokens=10)
    except ValueError as exc:
        assert "max_tokens" in str(exc)
    else:
        raise AssertionError("Expected invalid segmentation bounds to fail")


def test_loader_assigns_parent_and_segment_provenance():
    text = _long_sentence("Diagnose") + "\n\n" + _long_sentence("Next")
    session = Session(
        session_id="s1",
        file_path="s1.jsonl",
        messages=[
            Message(
                uuid="m1",
                role="assistant",
                content_blocks=[ContentBlock(block_type="thinking", text=text)],
            )
        ],
    )
    spans = segment_spans(
        session,
        SegmentationConfig(enabled=True, min_tokens=4, max_tokens=35),
    )
    assert len(spans) >= 2
    assert all(span.phase == SpanPhase.REASONING for span in spans)
    assert all(span.parent_block_id == "m1:0" for span in spans)
    assert [span.segment_index for span in spans] == list(range(len(spans)))
    assert [span.position for span in spans] == list(range(len(spans)))
    assert all(text[span.char_start : span.char_end] == span.text for span in spans)


def test_tool_calls_remain_atomic_when_fine_segmentation_is_enabled():
    text = "command output " * 200
    session = Session(
        session_id="s1",
        file_path="s1.jsonl",
        messages=[
            Message(
                uuid="m1",
                role="assistant",
                content_blocks=[
                    ContentBlock(
                        block_type="tool_use",
                        text=text,
                        tool_name="Bash",
                        tool_input={"command": "pytest"},
                    )
                ],
            )
        ],
    )
    spans = segment_spans(
        session,
        SegmentationConfig(enabled=True, min_tokens=4, max_tokens=20),
    )
    assert len(spans) == 1
    assert spans[0].tool_name == "Bash"
    assert spans[0].parent_block_id == "m1:0"


def test_short_mixed_purpose_paragraphs_can_split_below_maximum():
    text = (
        "The failure comes from an expired token and needs a refresh.\n\n"
        "As mentioned earlier, the token flow uses the same three steps.\n\n"
        "Next, update the refresh handler and rerun the authentication tests."
    )
    segments = segment_text(
        text,
        SegmentationConfig(enabled=True, min_tokens=5, max_tokens=100),
    )
    assert len(segments) == 3
    assert segments[0].text.startswith("The failure")
    assert segments[1].text.startswith("As mentioned")
    assert segments[2].text.startswith("Next")
