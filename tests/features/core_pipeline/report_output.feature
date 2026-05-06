Feature: TER Report Output
  As a developer who has completed a TER analysis
  I want to output the results in human-readable and machine-readable formats
  So that I can review, share, and integrate the results into my workflow

  The report generator supports two output formats:
  - Human-readable text with scores, phase breakdown, waste patterns, and token counts
  - Structured JSON with session_id, aggregate_ter, raw_ratio, phase_scores, tokens, and waste_patterns

  # ── Scenario 1: Human-readable text report ──────────────────────────────

  Scenario: Text report includes all required sections
    Given a completed TER calculation with the following results:
      | field          | value          |
      | session_id     | session-abc123 |
      | aggregate_ter  | 0.7234         |
      | raw_ratio      | 0.6891         |
      | total_tokens   | 12450          |
      | aligned_tokens | 8579           |
      | waste_tokens   | 3871           |
    And phase scores of reasoning=0.81, tool_use=0.65, generation=0.72
    And no waste patterns detected
    When the report is generated in text format
    Then the text output should contain the session identifier "session-abc123"
    And the text output should contain the aggregate TER score "0.7234"
    And the text output should contain a phase scores section with reasoning, tool_use, and generation
    And the text output should contain a token summary with total, aligned, and waste counts

  # ── Scenario 2: JSON report with all required keys ──────────────────────

  Scenario: JSON report contains all required fields
    Given a completed TER calculation with the following results:
      | field          | value          |
      | session_id     | session-abc123 |
      | aggregate_ter  | 0.7234         |
      | raw_ratio      | 0.6891         |
      | total_tokens   | 12450          |
      | aligned_tokens | 8579           |
      | waste_tokens   | 3871           |
    And phase scores of reasoning=0.81, tool_use=0.65, generation=0.72
    And a waste pattern of type "reasoning_loop" wasting 847 tokens
    When the report is generated in JSON format
    Then the JSON output should contain the key "session_id" with value "session-abc123"
    And the JSON output should contain the key "aggregate_ter" as a float between 0.0 and 1.0
    And the JSON output should contain the key "raw_ratio" as a float between 0.0 and 1.0
    And the JSON output should contain the key "phase_scores" with keys "reasoning", "tool_use", and "generation"
    And the JSON output should contain the key "total_tokens" as an integer
    And the JSON output should contain the key "aligned_tokens" as an integer
    And the JSON output should contain the key "waste_tokens" as an integer
    And the JSON output should contain the key "waste_patterns" as a list
    And total_tokens should equal aligned_tokens plus waste_tokens

  # ── Scenario 3: JSON round-trip validation ──────────────────────────────

  Scenario: JSON output can be parsed and values round-trip correctly
    Given a completed TER calculation with aggregate TER 0.7234 and total tokens 12450
    When the report is generated in JSON format
    And the JSON output is parsed back into a data structure
    Then the parsed aggregate_ter should be a float equal to 0.7234
    And the parsed total_tokens should equal the parsed aligned_tokens plus parsed waste_tokens
    And all phase scores should be floats between 0.0 and 1.0
    And the waste_patterns should be a valid list

  # ── Scenario 4: Text report includes waste pattern summaries ─────────────

  Scenario: Text report includes waste pattern summaries when patterns are detected
    Given a completed TER calculation with the following waste patterns:
      | type                | spans_involved | tokens_wasted | description                              |
      | reasoning_loop      | 3              | 847           | 3 consecutive redundant reasoning spans  |
      | duplicate_tool_call | 2              | 312           | Bash("ls -la"), repeated call             |
    When the report is generated in text format
    Then the text output should contain a waste patterns section
    And the text output should list 2 waste patterns
    And the text output should include "reasoning_loop" with 847 tokens wasted
    And the text output should include "duplicate_tool_call" with 312 tokens wasted
