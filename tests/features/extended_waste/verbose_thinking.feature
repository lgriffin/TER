Feature: Verbose Thinking Detection
  As a TER analyst
  I want to detect when an agent spends disproportionately many tokens thinking relative to acting
  So that verbose thinking waste is identified and quantified

  Background:
    Given the verbose thinking detector with default ratio_threshold of 10.0 and min_thinking_tokens of 500

  Scenario: Detect verbose thinking when ratio exceeds threshold
    Given a session with the following spans:
      | position | phase     | block_type | text                              | token_count |
      | 1        | reasoning | thinking   | Let me analyse this problem...    | 6000        |
      | 2        | tool_use  | tool_use   | Bash {"command":"ls"}             | 500         |
    When I run the verbose thinking detector
    Then 1 verbose thinking pattern should be detected
    And the pattern should report a ratio of 12.0
    And the tokens_wasted should be 1000
    # Because the excess is 6000 - (500 * 10.0) = 1000 tokens beyond the acceptable threshold

  Scenario: Below minimum thinking tokens is not flagged regardless of ratio
    Given a session with the following spans:
      | position | phase     | block_type | text                              | token_count |
      | 1        | reasoning | thinking   | Quick thought                     | 400         |
      | 2        | tool_use  | tool_use   | Bash {"command":"echo hi"}        | 10          |
    When I run the verbose thinking detector
    Then 0 verbose thinking patterns should be detected
    # Because the thinking block of 400 tokens is below the min_thinking_tokens threshold of 500
    # Even though the ratio of 40.0 exceeds the ratio_threshold of 10.0

  Scenario: Acceptable ratio is not flagged
    Given a session with the following spans:
      | position | phase     | block_type | text                              | token_count |
      | 1        | reasoning | thinking   | Detailed analysis of the codebase | 4000        |
      | 2        | tool_use  | tool_use   | Edit {"file_path":"/src/fix.py"}  | 500         |
    When I run the verbose thinking detector
    Then 0 verbose thinking patterns should be detected
    # Because the ratio of 8.0 is within the acceptable ratio_threshold of 10.0

  Scenario: Thinking with no subsequent action is always flagged
    Given a session with the following spans:
      | position | phase     | block_type | text                              | token_count |
      | 1        | reasoning | thinking   | Extensive deliberation on options  | 2000        |
    When I run the verbose thinking detector
    Then 1 verbose thinking pattern should be detected
    And the tokens_wasted should be 2000
    And the pattern should report 0 action tokens and infinite ratio
    # Because thinking with no subsequent action is pure waste

  Scenario: detect_all_extended runs all 5 detectors and sorts by start_position
    Given a session containing waste patterns from all 5 extended detectors:
      | position | phase     | block_type  | text                                          | token_count |
      | 1        | reasoning | thinking    | Very long deliberation block                  | 3000        |
      | 2        | tool_use  | tool_use    | Bash {"command":"echo x"}                     | 100         |
      | 5        | tool_use  | tool_use    | Read {"file_path":"/src/data.py"}             | 200         |
      | 6        | tool_use  | tool_use    | Read {"file_path":"/src/data.py"}             | 200         |
      | 7        | tool_use  | tool_use    | Read {"file_path":"/src/data.py"}             | 200         |
      | 10       | tool_use  | tool_use    | Bash {"command":"deploy --prod"}              | 50          |
      | 11       | tool_use  | tool_result | permission denied: production deploy locked   | 30          |
      | 12       | tool_use  | tool_use    | Bash {"command":"deploy --prod"}              | 50          |
      | 13       | tool_use  | tool_result | permission denied: production deploy locked   | 30          |
      | 14       | tool_use  | tool_use    | Bash {"command":"deploy --prod"}              | 50          |
    When I run detect_all_extended with default parameters
    Then patterns from multiple detectors should be returned
    And the results should be sorted by start_position in ascending order
    And the combined list should include verbose_thinking, over_reading, and permission_loop patterns
