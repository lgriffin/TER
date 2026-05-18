Feature: Rolling TER Computation
  As a developer monitoring Claude Code sessions in real time
  I want the TER calculator to maintain incremental rolling state
  So that efficiency signals are emitted accurately per assistant message

  Background:
    Given a fresh RollingTERState
    And a user message "Refactor the authentication module" has been processed

  Scenario: One TERSignal emitted per assistant message
    When the following assistant messages are processed:
      | text                                          |
      | I will refactor the auth module now            |
      | Here is the updated authentication code        |
      | I have also added tests for the auth module    |
    Then exactly 3 TERSignal objects are returned
    And each signal has an incremented message_index starting from 1

  Scenario: User messages update intent via exponential moving average
    Given a fresh RollingTERState
    When a user message "Fix the login bug" is processed
    And a user message "Also update the password reset flow" is processed
    Then the intent_embedding has shifted toward the second prompt
    And the intent is not a concatenated single embedding

  Scenario: Rolling state accumulates token totals correctly
    When an assistant message with 100 aligned tokens and 20 waste tokens is processed
    And another assistant message with 50 aligned tokens and 30 waste tokens is processed
    Then the state total_tokens equals 200
    And the state aligned_tokens equals 150
    And the state waste_tokens equals 50
    And total_tokens equals aligned_tokens plus waste_tokens

  Scenario: Duplicate request IDs are deduplicated with first-entry-wins
    When an assistant message with requestId "req-001" is processed
    And another assistant message with requestId "req-001" is processed
    Then only 1 TERSignal is returned
    And the state message_count is 1

  Scenario: Phase weights applied in aggregate TER calculation
    Given the phase weights are reasoning=0.3, tool_use=0.4, generation=0.3
    When an assistant message produces phase scores:
      | phase      | aligned | total |
      | reasoning  | 80      | 100   |
      | tool_use   | 60      | 100   |
      | generation | 90      | 100   |
    Then the aggregate TER equals 0.3*0.8 + 0.4*0.6 + 0.3*0.9 = 0.75

  Scenario: Phases with zero tokens default to score 1.0
    When an assistant message contributes tokens only to the generation phase
    And the reasoning phase has 0 total tokens
    And the tool_use phase has 0 total tokens
    Then the reasoning phase score defaults to 1.0
    And the tool_use phase score defaults to 1.0
    And the aggregate TER includes the default scores weighted at 0.3 and 0.4

  Scenario: User tool_result blocks are tracked for dedup but excluded from TER phase totals
    When a user message contains a tool_result block with content "file contents here"
    Then the tool_use phase totals are unchanged
    And the state span_count is incremented by 1
    And the total_tokens and aligned_tokens are unchanged
