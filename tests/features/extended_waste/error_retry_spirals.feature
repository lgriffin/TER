Feature: Error-Retry Spiral Detection
  As a TER analyst
  I want to detect when an agent retries a failed tool call with minimal parameter changes
  So that error-retry spiral waste is identified and quantified

  Background:
    Given the error-retry spiral detector with default similarity_threshold of 0.90 and min_retries of 3

  Scenario: Detect spiral with 3 retries and highly similar parameters
    Given a session with the following spans:
      | position | phase    | block_type  | text                                                         | token_count |
      | 1        | tool_use | tool_use    | Bash {"command":"python setup.py install --user"}            | 80          |
      | 2        | tool_use | tool_result | error: compilation failed in module core                     | 40          |
      | 3        | tool_use | tool_use    | Bash {"command":"python setup.py install --user"}            | 80          |
      | 4        | tool_use | tool_result | error: compilation failed in module core                     | 40          |
      | 5        | tool_use | tool_use    | Bash {"command":"python setup.py install --user"}            | 80          |
      | 6        | tool_use | tool_result | exception raised during build step                           | 40          |
      | 7        | tool_use | tool_use    | Bash {"command":"python setup.py install --user"}            | 80          |
    When I run the error-retry spiral detector
    Then 1 error-retry spiral pattern should be detected
    And the pattern should have tool_name "Bash"
    And the pattern should report 3 retries
    And the tokens_wasted should be 240
    And the tokens_wasted should exclude the initial attempt of 80 tokens

  Scenario: No spiral when parameters change significantly below 0.90 similarity
    Given a session with the following spans:
      | position | phase    | block_type  | text                                                         | token_count |
      | 1        | tool_use | tool_use    | Bash {"command":"python setup.py install --user"}            | 80          |
      | 2        | tool_use | tool_result | failed: missing dependency numpy                             | 40          |
      | 3        | tool_use | tool_use    | Bash {"command":"pip install numpy && python -m build"}      | 80          |
      | 4        | tool_use | tool_result | traceback in build process                                   | 40          |
      | 5        | tool_use | tool_use    | Bash {"command":"conda create -n env python=3.11 numpy"}    | 80          |
      | 6        | tool_use | tool_result | error: conda not found in PATH                              | 40          |
      | 7        | tool_use | tool_use    | Bash {"command":"apt-get install -y conda"}                  | 80          |
    When I run the error-retry spiral detector
    Then 0 error-retry spiral patterns should be detected
    # Because the trigram cosine similarity between consecutive parameters is below 0.90

  Scenario: No spiral with only 2 retries
    Given a session with the following spans:
      | position | phase    | block_type  | text                                                         | token_count |
      | 1        | tool_use | tool_use    | Bash {"command":"make build"}                                | 60          |
      | 2        | tool_use | tool_result | error: missing target                                        | 30          |
      | 3        | tool_use | tool_use    | Bash {"command":"make build"}                                | 60          |
      | 4        | tool_use | tool_result | error: missing target                                        | 30          |
      | 5        | tool_use | tool_use    | Bash {"command":"make build"}                                | 60          |
    When I run the error-retry spiral detector
    Then 0 error-retry spiral patterns should be detected
    # Because 2 retries is below the min_retries threshold of 3
