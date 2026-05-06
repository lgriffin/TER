Feature: Abandoned Approach Detection
  As a TER analyst
  I want to detect when an agent edits a file and then moves on without ever revisiting it
  So that abandoned approach waste is identified and quantified

  Scenario: File edited then never revisited while agent works on other files
    Given a session with the following spans:
      | position | phase    | block_type | text                                                | token_count |
      | 1        | tool_use | tool_use   | Edit {"file_path":"/src/attempt.py"}                | 150         |
      | 2        | tool_use | tool_use   | Write {"file_path":"/src/attempt.py"}               | 200         |
      | 3        | tool_use | tool_use   | Edit {"file_path":"/src/final.py"}                  | 180         |
      | 4        | tool_use | tool_use   | Read {"file_path":"/src/final.py"}                  | 100         |
    When I run the abandoned approach detector
    Then 1 abandoned approach pattern should be detected
    And the pattern should report file_path "/src/attempt.py"
    And the tokens_wasted should cover all spans that touched "/src/attempt.py"
    And the pattern description should indicate the file was edited but never revisited

  Scenario: File edited and revisited later is not flagged
    Given a session with the following spans:
      | position | phase    | block_type | text                                                | token_count |
      | 1        | tool_use | tool_use   | Edit {"file_path":"/src/module.py"}                 | 150         |
      | 2        | tool_use | tool_use   | Edit {"file_path":"/src/other.py"}                  | 120         |
      | 3        | tool_use | tool_use   | Read {"file_path":"/src/module.py"}                 | 100         |
      | 4        | tool_use | tool_use   | Read {"file_path":"/src/other.py"}                  | 100         |
    When I run the abandoned approach detector
    Then 0 abandoned approach patterns should be detected
    # Because both files were revisited after editing

  Scenario: Last file in session is not flagged as abandoned
    Given a session with the following spans:
      | position | phase    | block_type | text                                                | token_count |
      | 1        | tool_use | tool_use   | Edit {"file_path":"/src/first.py"}                  | 100         |
      | 2        | tool_use | tool_use   | Read {"file_path":"/src/first.py"}                  | 80          |
      | 3        | tool_use | tool_use   | Write {"file_path":"/src/final.py"}                 | 200         |
    When I run the abandoned approach detector
    Then the file "/src/final.py" should not be flagged as abandoned
    # Because it is the last file touched in the session with no subsequent work on other files
