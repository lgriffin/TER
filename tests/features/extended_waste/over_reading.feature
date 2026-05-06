Feature: Over-Reading Detection
  As a TER analyst
  I want to detect when an agent reads the same file repeatedly without writing to it
  So that redundant file-reading waste is identified and quantified

  Background:
    Given the over-reading detector with default min_reads of 2

  Scenario: File read 3 times without intervening write is detected
    Given a session with the following spans:
      | position | phase    | block_type | text                                               | token_count |
      | 1        | tool_use | tool_use   | Read {"file_path":"/src/main.py"}                  | 200         |
      | 2        | tool_use | tool_use   | Read {"file_path":"/src/main.py"}                  | 200         |
      | 3        | tool_use | tool_use   | Read {"file_path":"/src/main.py"}                  | 200         |
    When I run the over-reading detector
    Then 1 over-reading pattern should be detected
    And the pattern should report 3 total reads and 2 redundant reads
    And the tokens_wasted should be 400
    And the tokens_wasted should exclude the first legitimate read of 200 tokens

  Scenario: Write to a file resets its read counter
    Given a session with the following spans:
      | position | phase    | block_type | text                                               | token_count |
      | 1        | tool_use | tool_use   | Read {"file_path":"/src/config.py"}                | 150         |
      | 2        | tool_use | tool_use   | Read {"file_path":"/src/config.py"}                | 150         |
      | 3        | tool_use | tool_use   | Edit {"file_path":"/src/config.py"}                | 100         |
      | 4        | tool_use | tool_use   | Read {"file_path":"/src/config.py"}                | 150         |
      | 5        | tool_use | tool_use   | Read {"file_path":"/src/config.py"}                | 150         |
    When I run the over-reading detector
    Then 0 over-reading patterns should be detected
    # Because the Edit at position 3 resets the read tracker for that file
    And the subsequent 2 reads after the Edit give only 1 redundant read which is below min_reads of 2

  Scenario: Different files are tracked independently
    Given a session with the following spans:
      | position | phase    | block_type | text                                               | token_count |
      | 1        | tool_use | tool_use   | Read {"file_path":"/src/alpha.py"}                 | 100         |
      | 2        | tool_use | tool_use   | Read {"file_path":"/src/beta.py"}                  | 120         |
      | 3        | tool_use | tool_use   | Read {"file_path":"/src/alpha.py"}                 | 100         |
      | 4        | tool_use | tool_use   | Read {"file_path":"/src/beta.py"}                  | 120         |
      | 5        | tool_use | tool_use   | Read {"file_path":"/src/alpha.py"}                 | 100         |
      | 6        | tool_use | tool_use   | Read {"file_path":"/src/beta.py"}                  | 120         |
    When I run the over-reading detector
    Then 2 over-reading patterns should be detected
    And a pattern for "/src/alpha.py" should report 3 reads and 2 redundant reads
    And a pattern for "/src/beta.py" should report 3 reads and 2 redundant reads

  Scenario: Exactly 2 reads of the same file is not flagged
    Given a session with the following spans:
      | position | phase    | block_type | text                                               | token_count |
      | 1        | tool_use | tool_use   | Read {"file_path":"/src/utils.py"}                 | 180         |
      | 2        | tool_use | tool_use   | Read {"file_path":"/src/utils.py"}                 | 180         |
    When I run the over-reading detector
    Then 0 over-reading patterns should be detected
    # Because 2 total reads means only 1 redundant read which is below the min_reads threshold of 2
