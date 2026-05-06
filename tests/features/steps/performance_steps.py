"""Step definitions for performance and acceleration features."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pytest_bdd import given, when, then, parsers, scenarios

from ter_calculator.acceleration import (
    AnalysisCache,
    QuickAnalyser,
    SessionWatcher,
    WatchEvent,
    WatchEventType,
)
from ter_calculator.token_counting import (
    CountMethod,
    calibrate_multiplier,
    count_tokens,
    estimate_tokens_heuristic,
    token_count_confidence,
)

scenarios(
    "../performance_acceleration/token_counting.feature",
    "../performance_acceleration/analysis_cache.feature",
    "../performance_acceleration/quick_analyser.feature",
    "../performance_acceleration/session_watcher.feature",
)


@pytest.fixture
def context():
    return {}


# ===========================================================================
# Token counting -- token_counting.feature
# ===========================================================================


@given(parsers.parse("a text of {chars:d} characters"))
def text_chars(context, chars):
    context["text"] = "a" * chars


@given(parsers.parse('the phase is "{phase}"'))
def set_phase(context, phase):
    context["phase"] = phase


@given("no phase is specified")
def no_phase(context):
    context["phase"] = None


@when("tokens are estimated via heuristic")
def estimate_heuristic(context):
    context["result"] = estimate_tokens_heuristic(
        context["text"], phase=context.get("phase")
    )


@then(parsers.parse("the result is approximately {expected:d}"))
def check_approx(context, expected):
    assert context["result"] == pytest.approx(expected, abs=5)


@then(parsers.parse("the result is {expected:d}"))
def check_exact(context, expected):
    assert context["result"] == expected


# -- Calibration --

@given("calibration samples with known text and token count pairs")
def calibration_samples(context):
    context["samples"] = [
        ("hello world test", 3),
        ("The quick brown fox jumps over the lazy dog", 9),
        ("import os", 2),
    ]


@when("calibrate_multiplier is called")
def call_calibrate(context):
    try:
        context["multiplier"] = calibrate_multiplier(context.get("samples", []))
    except ValueError as e:
        context["error"] = e


@then("a positive float multiplier is returned")
def check_positive_multiplier(context):
    assert context["multiplier"] > 0.0


@given(parsers.parse("a calibrated multiplier of {m:f}"))
def set_calibrated(context, m):
    context["calibrated_multiplier"] = m


@when("count_tokens is called with the calibrated multiplier")
def call_count_calibrated(context):
    context["count_result"] = count_tokens(
        context["text"],
        calibrated_multiplier=context["calibrated_multiplier"],
    )


@then(parsers.parse('method_used is "{method}"'))
def check_method(context, method):
    assert context["count_result"].method_used.value == method


@then(parsers.parse("confidence is approximately {c:f}"))
def check_confidence_approx(context, c):
    assert context["count_result"].confidence == pytest.approx(c, abs=0.15)


# -- Heuristic confidence --

@given("a natural-language text")
def natural_language_text(context):
    context["text"] = (
        "The quick brown fox jumps over the lazy dog and runs through the meadow "
        "while the sun shines brightly overhead casting long shadows on the ground"
    )


@when("count_tokens is called via heuristic")
def call_count_via_heuristic(context):
    context["count_result"] = count_tokens(context["text"])


# -- Code-heavy confidence --

@given("a text with many structural punctuation characters like braces and semicolons")
def code_heavy_text(context):
    context["text"] = (
        'if (x) { y = [1, 2, 3]; z = {a: 1}; } else { w = (a < b) ? c : d; } '
        'for (i = 0; i < 10; i++) { arr[i] = fn(i); } switch(x) { case 1: break; }'
    )


@when("token_count_confidence is computed for heuristic method")
def compute_confidence_heuristic(context):
    conf = token_count_confidence(context["text"], CountMethod.HEURISTIC)
    # Store on a synthetic count_result so the "confidence is below" step can read it
    context["computed_confidence"] = conf


@then("confidence is below 0.8")
def check_confidence_below(context):
    assert context["computed_confidence"] < 0.8


# -- API counting --

@given("the Anthropic API is available")
def anthropic_api_available(context):
    # We mock the API call to avoid needing a real key in tests
    context["text"] = "Hello, world! This is a test of API token counting."
    context["mock_api_tokens"] = 12


@when("count_tokens is called with use_api enabled")
def call_count_with_api(context):
    mock_response = MagicMock()
    mock_response.input_tokens = context["mock_api_tokens"]

    mock_client = MagicMock()
    mock_client.messages.count_tokens.return_value = mock_response

    with patch(
        "ter_calculator.token_counting.anthropic", create=True
    ) as mock_anthropic_mod:
        mock_anthropic_mod.Anthropic.return_value = mock_client
        # Patch the lazy import inside _count_tokens_via_api
        import importlib
        import ter_calculator.token_counting as tc_mod

        original_func = tc_mod._count_tokens_via_api

        def patched_api(text):
            try:
                client = mock_client
                response = client.messages.count_tokens(
                    model="claude-sonnet-4-20250514",
                    messages=[{"role": "user", "content": text}],
                )
                return response.input_tokens
            except Exception:
                return None

        tc_mod._count_tokens_via_api = patched_api
        try:
            context["count_result"] = count_tokens(
                context["text"], use_api=True
            )
        finally:
            tc_mod._count_tokens_via_api = original_func


@then("confidence is 1.0")
def check_confidence_exact_one(context):
    assert context["count_result"].confidence == 1.0


# -- Empty text --

@given("an empty text string")
def empty_text(context):
    context["text"] = ""


@when("count_tokens is called")
def call_count_tokens(context):
    context["count_result"] = count_tokens(
        context.get("text", ""),
        phase=context.get("phase"),
        calibrated_multiplier=context.get("calibrated_multiplier"),
    )


@then(parsers.parse("estimated_tokens is {n:d}"))
def check_estimated(context, n):
    assert context["count_result"].estimated_tokens == n


# -- Empty samples --

@given("an empty list of calibration samples")
def empty_samples(context):
    context["samples"] = []


@then("a ValueError is raised")
def check_value_error(context):
    assert isinstance(context.get("error"), ValueError)


# ===========================================================================
# Analysis cache -- analysis_cache.feature
# ===========================================================================


@given("a temporary cache directory")
def cache_dir(tmp_path, context):
    context["cache"] = AnalysisCache(cache_dir=tmp_path / "cache")


@given(parsers.parse("the default cache TTL is {hours:d} hours"))
def cache_ttl(context, hours):
    context["cache_ttl"] = hours


# -- Cache miss --

@given("a cache key that has not been stored")
def cache_miss_key(context):
    context["cache_key"] = "miss-key-abc123"
    context["compute_called"] = False

    def compute():
        context["compute_called"] = True
        return {"result": 42}

    context["compute_fn"] = compute


@when("get_or_compute is called with a compute function")
def call_get_or_compute(context):
    context["cached_value"] = context["cache"].get_or_compute(
        context["cache_key"], context["compute_fn"]
    )


@then("the compute function is invoked")
def check_compute_called(context):
    assert context["compute_called"] is True


@then("the result is stored in the cache")
def check_stored(context):
    assert context["cached_value"] is not None


@then(parsers.parse("cache_stats reports miss_count of {n:d}"))
def check_miss_count(context, n):
    stats = context["cache"].cache_stats()
    assert stats.miss_count == n


# -- Cache hit --

@given("a cache key with a previously stored result")
def cache_hit_key(context):
    key = "hit-key-def456"
    context["cache"].get_or_compute(key, lambda: {"result": 99})
    context["cache_key"] = key
    context["compute_called"] = False
    context["compute_fn"] = lambda: (_ for _ in ()).throw(
        AssertionError("should not be called")
    )


@when("get_or_compute is called with the same key")
def call_get_or_compute_again(context):
    context["cached_value"] = context["cache"].get_or_compute(
        context["cache_key"], context["compute_fn"]
    )


@then("the compute function is not invoked")
def check_compute_not_called(context):
    assert context.get("compute_called", False) is False


@then("the previously stored result is returned")
def check_cached_result(context):
    assert context["cached_value"] == {"result": 99}


@then(parsers.parse("cache_stats reports hit_count of {n:d}"))
def check_hit_count(context, n):
    stats = context["cache"].cache_stats()
    assert stats.hit_count == n


# -- Expired cache entries --

@given("a cache entry older than the TTL")
def expired_cache_entry(context):
    cache = context["cache"]
    key = "expired-key-xyz789"
    context["cache_key"] = key
    context["compute_called"] = False

    # Store a value first
    cache.get_or_compute(key, lambda: {"old": True}, ttl_hours=1)

    # Manually backdate the metadata timestamp so it is expired
    pkl_path, meta_path = cache._key_paths(key)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # Set timestamp to 2 hours ago (TTL is 1 hour)
        meta["timestamp"] = time.time() - 7200
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def compute():
        context["compute_called"] = True
        return {"new": True}

    context["compute_fn"] = compute
    # Use a short TTL so the backdated entry is expired
    context["cache_ttl_override"] = 1


@when("get_or_compute is called for that key")
def call_get_or_compute_for_key(context):
    ttl = context.get("cache_ttl_override", context.get("cache_ttl", 168))
    context["cached_value"] = context["cache"].get_or_compute(
        context["cache_key"], context["compute_fn"], ttl_hours=ttl
    )


@then("the entry is treated as a miss")
def check_treated_as_miss(context):
    # The compute function was called (confirming it was a miss)
    assert context["compute_called"] is True


# -- Invalidate --

@given("cached results for a session file")
def cached_results_for_session(tmp_path, context):
    cache = context["cache"]
    # Create a dummy session file
    session_file = tmp_path / "session_to_invalidate.jsonl"
    session_file.write_text('{"sessionId": "test"}\n', encoding="utf-8")
    context["session_path_to_invalidate"] = str(session_file)

    # Store a cache entry using the session file's hash as the key
    from ter_calculator.acceleration import hash_file
    file_hash = hash_file(session_file)
    cache.get_or_compute(file_hash, lambda: {"cached": True})

    # Verify it was stored
    stats = cache.cache_stats()
    assert stats.entry_count >= 1
    context["invalidation_key"] = file_hash


@when("invalidate is called for that session path")
def call_invalidate(context):
    context["cache"].invalidate(context["session_path_to_invalidate"])


@then("the cached entries for that session are removed")
def check_entries_removed(context):
    # After invalidation, trying to read the key should return None
    cache = context["cache"]
    result = cache._read(context["invalidation_key"], ttl_hours=168)
    assert result is None


# -- Clear all --

@given("a cache with multiple entries")
def cache_with_multiple_entries(context):
    cache = context["cache"]
    cache.get_or_compute("multi-key-1", lambda: {"a": 1})
    cache.get_or_compute("multi-key-2", lambda: {"b": 2})
    cache.get_or_compute("multi-key-3", lambda: {"c": 3})
    stats = cache.cache_stats()
    assert stats.entry_count >= 3


@when("clear_all is called")
def call_clear_all(context):
    context["cache"].clear_all()


@then(parsers.parse("cache_stats reports entry_count of {n:d}"))
def check_entry_count(context, n):
    stats = context["cache"].cache_stats()
    assert stats.entry_count == n


@then("hit_count and miss_count are reset to 0")
def check_counts_reset(context):
    stats = context["cache"].cache_stats()
    assert stats.hit_count == 0
    assert stats.miss_count == 0


# ===========================================================================
# Quick analyser -- quick_analyser.feature
# ===========================================================================


@given(parsers.parse("a QuickAnalyser with top_n_keywords {n:d}"))
def quick_analyser(context, n):
    context["analyser"] = QuickAnalyser(top_n_keywords=n)


@given("a valid session JSONL file")
def valid_session(tmp_path, context):
    path = tmp_path / "session.jsonl"
    lines = [
        {
            "sessionId": "qs",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "add login page"}
                ],
            },
        },
        {
            "sessionId": "qs",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Creating login page with authentication.",
                    }
                ],
            },
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    context["session_path"] = str(path)


@when("quick analysis is run")
def run_quick(context):
    try:
        context["quick_result"] = context["analyser"].analyse_quick(
            context["session_path"]
        )
    except FileNotFoundError as e:
        context["error"] = e


@then("the result contains session_id and aggregate_ter")
def check_quick_keys(context):
    r = context["quick_result"]
    assert "session_id" in r
    assert "aggregate_ter" in r


@then(parsers.parse('the result method is "{method}"'))
def check_quick_method(context, method):
    assert context["quick_result"]["method"] == method


@then("total_tokens equals aligned_tokens plus waste_tokens")
def check_quick_invariant(context):
    r = context["quick_result"]
    assert r["total_tokens"] == r["aligned_tokens"] + r["waste_tokens"]


# -- Keywords exclude stop words --

@given(
    parsers.parse(
        'user prompts mentioning "{kw1}", "{kw2}", and "{kw3}"'
    )
)
def user_prompts_with_keywords(context, kw1, kw2, kw3):
    context["prompt_text"] = f"Please fix the {kw1} and {kw2} issues. Also {kw3} system needs work."
    context["expected_included"] = {kw1.lower(), kw2.lower()}
    context["expected_excluded"] = {kw3.lower()}


@when("keywords are extracted")
def extract_keywords(context):
    analyser = context["analyser"]
    context["keyword_set"] = analyser._extract_keywords(
        [context["prompt_text"]]
    )


@then(
    parsers.parse(
        'the keyword set includes "{kw1}" and "{kw2}"'
    )
)
def check_keywords_included(context, kw1, kw2):
    ks = context["keyword_set"]
    assert kw1.lower() in ks, f"{kw1} not found in {ks}"
    assert kw2.lower() in ks, f"{kw2} not found in {ks}"


@then(parsers.parse('the keyword set excludes "{word}"'))
def check_keywords_excluded(context, word):
    assert word.lower() not in context["keyword_set"]


# -- Empty session --

@given("a session JSONL file with no content spans")
def empty_session(tmp_path, context):
    path = tmp_path / "empty_session.jsonl"
    lines = [
        {"sessionId": "empty-sess"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    context["session_path"] = str(path)


@then(parsers.parse("aggregate_ter is {value:f}"))
def check_aggregate(context, value):
    assert context["quick_result"]["aggregate_ter"] == pytest.approx(
        value, abs=0.01
    )


# -- Only stop words --

@given("a session with user prompts containing only stop words")
def stop_words_session(tmp_path, context):
    path = tmp_path / "stopwords_session.jsonl"
    lines = [
        {
            "sessionId": "sw-sess",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "the a an is are was to be"}
                ],
            },
        },
        {
            "sessionId": "sw-sess",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a response that contains some output.",
                    }
                ],
            },
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    context["session_path"] = str(path)


# -- Keyword overlap --

@given(parsers.parse("a set of {n:d} keywords"))
def set_of_keywords(context, n):
    # Generate exactly n keywords
    context["keywords"] = {f"keyword{i}" for i in range(n)}


@given(parsers.parse("a span text containing {m:d} of those keywords"))
def span_with_some_keywords(context, m):
    keywords = sorted(context["keywords"])
    # Build text containing exactly m keywords
    selected = keywords[:m]
    context["span_text"] = " ".join(selected) + " some other words"


@when("keyword overlap is computed")
def compute_overlap(context):
    score = QuickAnalyser._keyword_overlap_score(
        context["span_text"], context["keywords"]
    )
    context["overlap_score"] = score


@then(parsers.parse("the score is {s:f}"))
def check_overlap_score(context, s):
    assert context["overlap_score"] == pytest.approx(s, abs=0.01)


# -- Missing session file --

@given("a path to a non-existent JSONL file")
def nonexistent_file(tmp_path, context):
    context["session_path"] = str(tmp_path / "nope.jsonl")


@then("a FileNotFoundError is raised")
def check_fnf(context):
    assert isinstance(context.get("error"), FileNotFoundError)


# -- Deduplication --

@given(
    "a session file with duplicate requestIds having different output_tokens"
)
def session_with_dupes(tmp_path, context):
    path = tmp_path / "dupes_session.jsonl"
    lines = [
        {
            "sessionId": "dup-sess",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "build the login feature"}],
            },
        },
        {
            "sessionId": "dup-sess",
            "requestId": "req-1",
            "message": {
                "role": "assistant",
                "requestId": "req-1",
                "content": [{"type": "text", "text": "First attempt at login."}],
                "usage": {"output_tokens": 50},
            },
        },
        {
            "sessionId": "dup-sess",
            "requestId": "req-1",
            "message": {
                "role": "assistant",
                "requestId": "req-1",
                "content": [
                    {
                        "type": "text",
                        "text": "Better login implementation with validation and error handling.",
                    }
                ],
                "usage": {"output_tokens": 200},
            },
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    context["session_path"] = str(path)


@when("quick analysis parses the session")
def parse_session_for_dedup(context):
    data = QuickAnalyser._parse_session(context["session_path"])
    context["parsed_data"] = data


@then(
    "only the entry with the highest output_tokens is kept per requestId"
)
def check_dedup(context):
    spans = context["parsed_data"]["spans"]
    # Only one span for the assistant role with requestId "req-1" should remain,
    # and it should be the one with higher output_tokens (the longer text).
    req1_spans = [
        s
        for s in spans
        if "Better login" in s["text"] or "First attempt" in s["text"]
    ]
    assert len(req1_spans) == 1
    assert "Better login" in req1_spans[0]["text"]


# ===========================================================================
# Session watcher -- session_watcher.feature
# ===========================================================================


@given("the default watch polling interval is 30 seconds")
def default_poll_interval(context):
    context["poll_interval"] = 30


# -- Detect new session file --

@given(parsers.parse("a watched directory with {n:d} existing JSONL files"))
def watched_dir_with_files(tmp_path, context, n):
    watch_dir = tmp_path / "watched"
    watch_dir.mkdir()
    for i in range(n):
        (watch_dir / f"existing_{i}.jsonl").write_text(
            json.dumps({"sessionId": f"s{i}"}) + "\n", encoding="utf-8"
        )
    context["watch_dir"] = watch_dir
    context["watcher"] = SessionWatcher()
    context["events"] = []


@when("a new JSONL file is added to the directory")
def add_new_jsonl(context):
    new_file = context["watch_dir"] / "new_session.jsonl"
    # Don't create it yet -- let the watcher snapshot first, then we create it
    context["new_file"] = new_file


@when("the watcher polls")
def watcher_polls(context):
    watcher = context["watcher"]
    watch_dir = context["watch_dir"]
    events = context["events"]

    def collect_event(event):
        events.append(event)

    # Build initial snapshot
    watcher._known_files = watcher._snapshot(watch_dir)

    # Now create the new file if pending
    if "new_file" in context and not context["new_file"].exists():
        context["new_file"].write_text(
            json.dumps({"sessionId": "new"}) + "\n", encoding="utf-8"
        )

    # If we need to update mtime, bump it forward so the poll detects a change
    if context.get("mtime_changed"):
        target = context["modified_file"]
        # Append content to change the file
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "modified", "extra": True}) + "\n")
        # Force mtime to be ahead of the snapshot value
        stat = target.stat()
        os.utime(target, (stat.st_atime, stat.st_mtime + 2))

    # Poll once
    watcher._poll(watch_dir, collect_event)


@then("a NEW_SESSION event is emitted for the new file")
def check_new_session_event(context):
    events = context["events"]
    new_events = [
        e for e in events if e.event_type == WatchEventType.NEW_SESSION
    ]
    assert len(new_events) >= 1
    assert any(
        "new_session" in e.file_path for e in new_events
    )


# -- Detect modified session file --

@given("a watched directory with an existing JSONL file")
def watched_dir_with_one_file(tmp_path, context):
    watch_dir = tmp_path / "watched_mod"
    watch_dir.mkdir()
    existing = watch_dir / "existing.jsonl"
    existing.write_text(
        json.dumps({"sessionId": "s0"}) + "\n", encoding="utf-8"
    )
    context["watch_dir"] = watch_dir
    context["modified_file"] = existing
    context["watcher"] = SessionWatcher()
    context["events"] = []


@when("the file modification time changes")
def change_mtime(context):
    context["mtime_changed"] = True


@then("a MODIFIED_SESSION event is emitted")
def check_modified_event(context):
    events = context["events"]
    mod_events = [
        e for e in events if e.event_type == WatchEventType.MODIFIED_SESSION
    ]
    assert len(mod_events) >= 1


# -- Callback invoked --

@given("a watcher with a registered callback")
def watcher_with_callback(tmp_path, context):
    watch_dir = tmp_path / "watched_cb"
    watch_dir.mkdir()
    context["watch_dir"] = watch_dir
    context["watcher"] = SessionWatcher()
    context["events"] = []
    context["callback_invoked"] = False

    def cb(event):
        context["events"].append(event)
        context["callback_invoked"] = True

    context["callback"] = cb


@when("a new session file appears and the watcher polls")
def new_file_and_poll(context):
    watcher = context["watcher"]
    watch_dir = context["watch_dir"]

    # Snapshot empty directory
    watcher._known_files = watcher._snapshot(watch_dir)

    # Add a new file
    new_file = watch_dir / "callback_test.jsonl"
    new_file.write_text(
        json.dumps({"sessionId": "cb"}) + "\n", encoding="utf-8"
    )

    # Poll with the callback
    watcher._poll(watch_dir, context["callback"])


@then("the callback receives a WatchEvent with the file path and timestamp")
def check_callback_event(context):
    assert context["callback_invoked"] is True
    events = context["events"]
    assert len(events) >= 1
    event = events[0]
    assert isinstance(event, WatchEvent)
    assert event.file_path  # non-empty
    assert event.timestamp > 0


# -- Stop terminates --

@given("a running watcher")
def running_watcher(tmp_path, context):
    watch_dir = tmp_path / "watched_stop"
    watch_dir.mkdir()
    context["watch_dir"] = watch_dir
    watcher = SessionWatcher()
    context["watcher"] = watcher
    context["watch_thread_finished"] = threading.Event()

    def run_watch():
        watcher.watch(
            project_path=str(watch_dir),
            interval=1,
        )
        context["watch_thread_finished"].set()

    t = threading.Thread(target=run_watch, daemon=True)
    t.start()
    context["watch_thread"] = t
    # Give the watcher a moment to start its loop
    time.sleep(0.3)


@when("stop is called")
def call_stop(context):
    context["watcher"].stop()


@then("the watcher exits its polling loop")
def check_watcher_stopped(context):
    finished = context["watch_thread_finished"].wait(timeout=5)
    assert finished, "Watcher did not stop within timeout"
