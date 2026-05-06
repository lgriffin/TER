Feature: Compare TER Across Sessions
  As a developer or team lead
  I want to compare TER scores across multiple Claude Code sessions
  So that I can identify trends and determine if prompt or workflow changes improve efficiency

  The comparison tool displays aggregate TER, phase scores, token counts, and waste
  pattern counts side by side, normalized for session length, and ranked by TER.

  Background:
    Given the default phase weights are reasoning=0.3, tool_use=0.4, generation=0.3

  # ── Scenario 1: Side-by-side comparison ──────────────────────────────────

  Scenario: Compare two sessions side by side
    Given a session "session-alpha" with aggregate TER 0.82 and the following details:
      | metric         | value |
      | total_tokens   | 5200  |
      | aligned_tokens | 4264  |
      | waste_tokens   | 936   |
      | reasoning      | 0.85  |
      | tool_use       | 0.78  |
      | generation     | 0.90  |
    And a session "session-beta" with aggregate TER 0.65 and the following details:
      | metric         | value |
      | total_tokens   | 12000 |
      | aligned_tokens | 7800  |
      | waste_tokens   | 4200  |
      | reasoning      | 0.70  |
      | tool_use       | 0.55  |
      | generation     | 0.75  |
    When the sessions are compared
    Then the comparison should include both sessions
    And each session should show aggregate TER, phase scores, total tokens, and waste tokens
    And "session-alpha" should have a higher TER than "session-beta"

  # ── Scenario 2: Normalized for session size ──────────────────────────────

  Scenario: Comparison is normalized for different session sizes
    Given a short session "session-small" with 1000 total tokens and TER 0.80
    And a long session "session-large" with 50000 total tokens and TER 0.80
    When the sessions are compared
    Then both sessions should have comparable TER scores despite different sizes
    And the comparison should not bias toward shorter or longer sessions

  # ── Scenario 3: Compare 10 or more sessions (SC-004) ────────────────────

  Scenario: Compare 10 or more sessions in a single invocation
    Given the following 12 sessions with TER scores:
      | session_id  | aggregate_ter |
      | session-01  | 0.92          |
      | session-02  | 0.88          |
      | session-03  | 0.85          |
      | session-04  | 0.81          |
      | session-05  | 0.77          |
      | session-06  | 0.73          |
      | session-07  | 0.69          |
      | session-08  | 0.65          |
      | session-09  | 0.60          |
      | session-10  | 0.55          |
      | session-11  | 0.48          |
      | session-12  | 0.42          |
    When all 12 sessions are compared in a single invocation
    Then the comparison should include all 12 sessions
    And the comparison should complete without error

  # ── Scenario 4: Ranking by TER ───────────────────────────────────────────

  Scenario: Sessions are ranked by TER score
    Given the following sessions with TER scores:
      | session_id   | aggregate_ter |
      | session-low  | 0.45          |
      | session-mid  | 0.72          |
      | session-high | 0.91          |
    When the sessions are compared with ranking by TER
    Then the sessions should be ranked in descending order of aggregate TER
    And the first ranked session should be "session-high" with TER 0.91
    And the second ranked session should be "session-mid" with TER 0.72
    And the third ranked session should be "session-low" with TER 0.45

  # ── Scenario 5: Single session produces a warning ────────────────────────

  Scenario: Single session comparison produces a warning
    Given only one session "session-solo" with aggregate TER 0.75
    When a comparison is requested with a single session
    Then the system should produce a warning that comparison requires multiple sessions
    And the single session result should still be displayed
