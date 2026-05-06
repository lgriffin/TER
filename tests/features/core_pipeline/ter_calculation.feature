Feature: TER Calculation for Completed Sessions
  As a developer who has just finished a Claude Code session
  I want to calculate a Token Efficiency Ratio for that session
  So that I understand how efficiently tokens were used toward my goal

  The TER calculator accepts session data and produces an aggregate TER score
  between 0.0 and 1.0, per-phase scores for reasoning, tool_use, and generation,
  and counts of total, aligned, and waste tokens.

  Default phase weights: reasoning=0.3, tool_use=0.4, generation=0.3
  Default similarity threshold: 0.40
  Default confidence threshold: 0.75
  Aggregate TER = sum(phase_weight * phase_score)
  Phase score = phase_aligned_tokens / phase_total_tokens (default 1.0 when phase has no tokens)
  total_tokens = aligned_tokens + waste_tokens

  Background:
    Given the default phase weights are reasoning=0.3, tool_use=0.4, generation=0.3
    And the default similarity threshold is 0.40
    And the default confidence threshold is 0.75

  # ── Scenario 1: Mixed alignment session ─────────────────────────────────

  Scenario: Mixed alignment session produces valid TER with phase breakdown
    Given a completed session with the following spans:
      | phase      | tokens | aligned |
      | reasoning  | 200    | 160     |
      | tool_use   | 300    | 210     |
      | generation | 100    | 80      |
    When the TER is calculated
    Then the aggregate TER should be between 0.0 and 1.0
    And the phase scores should be:
      | phase      | score |
      | reasoning  | 0.80  |
      | tool_use   | 0.70  |
      | generation | 0.80  |
    And the aggregate TER should be 0.76
    And total_tokens should equal 600
    And aligned_tokens should equal 450
    And waste_tokens should equal 150
    And total_tokens should equal aligned_tokens plus waste_tokens

  # ── Scenario 2: Perfect session ──────────────────────────────────────────

  Scenario: Perfectly aligned session produces TER of 1.0
    Given a completed session where all token spans are aligned to the intent
    When the TER is calculated
    Then the aggregate TER should be 1.0
    And waste_tokens should equal 0
    And total_tokens should equal aligned_tokens plus waste_tokens

  # ── Scenario 3: Fully wasteful session ──────────────────────────────────

  Scenario: Fully wasteful session produces TER of 0.0
    Given a completed session where no token spans are aligned to the intent
    When the TER is calculated
    Then the aggregate TER should be 0.0
    And aligned_tokens should equal 0
    And total_tokens should equal aligned_tokens plus waste_tokens

  # ── Scenario 4: Empty session ────────────────────────────────────────────

  Scenario: Empty session produces an error
    Given a session with no messages
    When the TER calculation is attempted
    Then the system should return an error indicating no session data is available

  # ── Scenario 5: Single message ───────────────────────────────────────────

  Scenario: Session with a single message is processable
    Given a session containing exactly one user prompt and one assistant response
    When the TER is calculated
    Then the aggregate TER should be between 0.0 and 1.0
    And the result should include phase scores for reasoning, tool_use, and generation
    And the result should include total_tokens, aligned_tokens, and waste_tokens

  # ── Scenario 6: Only tool calls ──────────────────────────────────────────

  Scenario: Session with only tool call tokens defaults other phases to 1.0
    Given a completed session with the following spans:
      | phase    | tokens | aligned |
      | tool_use | 500    | 350     |
    And no reasoning or generation tokens are present
    When the TER is calculated
    Then the phase scores should be:
      | phase      | score |
      | reasoning  | 1.0   |
      | tool_use   | 0.70  |
      | generation | 1.0   |
    And the aggregate TER should be 0.88

  # ── Scenario 7: Reproducible results (SC-002) ───────────────────────────

  Scenario: Same session input produces identical TER results
    Given a completed session with recorded interaction data
    When the TER is calculated two separate times on the same input
    Then both results should have identical aggregate TER scores
    And both results should have identical phase scores
    And both results should have identical token counts

  # ── Scenario 8: Custom phase weights ─────────────────────────────────────

  Scenario: Custom phase weights are applied correctly
    Given a completed session with the following spans:
      | phase      | tokens | aligned |
      | reasoning  | 200    | 200     |
      | tool_use   | 200    | 0       |
      | generation | 200    | 200     |
    And custom phase weights of reasoning=0.5, tool_use=0.2, generation=0.3
    When the TER is calculated
    Then the phase scores should be:
      | phase      | score |
      | reasoning  | 1.0   |
      | tool_use   | 0.0   |
      | generation | 1.0   |
    And the aggregate TER should be 0.80

  # ── Scenario 9: Phase score boundary validation ──────────────────────────

  Scenario Outline: Phase score equals aligned divided by total tokens
    Given a session phase "<phase>" with <aligned> aligned tokens out of <total> total tokens
    When the phase score is computed
    Then the phase score for "<phase>" should be <expected_score>

    Examples:
      | phase      | aligned | total | expected_score |
      | reasoning  | 100     | 100   | 1.0            |
      | tool_use   | 0       | 100   | 0.0            |
      | generation | 75      | 100   | 0.75           |
      | reasoning  | 0       | 0     | 1.0            |

  # ── Scenario 10: Large session within time limit (SC-005) ────────────────

  Scenario: Session with 100,000 tokens is processed within 120 seconds
    Given a completed session containing 100000 tokens across all phases
    When the TER is calculated
    Then the calculation should complete in under 120 seconds
    And the aggregate TER should be between 0.0 and 1.0

  # ── Scenario 11: Phase weights must sum to 1.0 ──────────────────────────

  Scenario: Phase weights that do not sum to 1.0 produce an error
    Given custom phase weights of reasoning=0.3, tool_use=0.3, generation=0.3
    When the TER calculation is attempted
    Then the system should return an error indicating phase weights must sum to 1.0
