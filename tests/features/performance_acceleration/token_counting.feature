Feature: Token Counting
  As a developer needing accurate token estimates
  I want phase-aware heuristic counting with optional calibration
  So that TER scores are based on reliable token counts

  Scenario Outline: Phase-aware heuristic uses correct multiplier
    Given a text of <char_count> characters
    And the phase is "<phase>"
    When tokens are estimated via heuristic
    Then the result is approximately <expected_tokens>

    Examples:
      | char_count | phase      | expected_tokens |
      | 400        | reasoning  | 100             |
      | 400        | generation | 100             |
      | 320        | tool_use   | 100             |

  Scenario: Default multiplier is 4.0 when no phase specified
    Given a text of 400 characters
    And no phase is specified
    When tokens are estimated via heuristic
    Then the result is 100

  Scenario: Calibration from sample data
    Given calibration samples with known text and token count pairs
    When calibrate_multiplier is called
    Then a positive float multiplier is returned

  Scenario: Calibrated counting uses the calibrated multiplier
    Given a calibrated multiplier of 3.8
    And a text of 380 characters
    When count_tokens is called with the calibrated multiplier
    Then estimated_tokens is 100
    And method_used is "calibrated"
    And confidence is approximately 0.9

  Scenario: Heuristic confidence is 0.8 baseline
    Given a natural-language text
    When count_tokens is called via heuristic
    Then confidence is approximately 0.8

  Scenario: Code-heavy text reduces heuristic confidence
    Given a text with many structural punctuation characters like braces and semicolons
    When token_count_confidence is computed for heuristic method
    Then confidence is below 0.8

  Scenario: API counting returns confidence of 1.0
    Given the Anthropic API is available
    When count_tokens is called with use_api enabled
    Then method_used is "api"
    And confidence is 1.0

  Scenario: Empty text returns 0 tokens
    Given an empty text string
    When count_tokens is called
    Then estimated_tokens is 0

  Scenario: Calibration with empty samples raises ValueError
    Given an empty list of calibration samples
    When calibrate_multiplier is called
    Then a ValueError is raised
