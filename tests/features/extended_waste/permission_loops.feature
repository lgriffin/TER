Feature: Permission Loop Detection
  As a TER analyst
  I want to detect when an agent retries a denied tool call unchanged
  So that permission loop waste is identified and quantified

  Background:
    Given the permission loop detector with default min_retries of 2

  Scenario: Detect permission loop with 2 retries after denial
    Given a session with the following spans:
      | position | phase    | block_type  | text                                          | token_count |
      | 1        | tool_use | tool_use    | Bash {"command":"rm -rf /root"}               | 50          |
      | 2        | tool_use | tool_result | permission denied: cannot remove /root        | 30          |
      | 3        | tool_use | tool_use    | Bash {"command":"rm -rf /root"}               | 50          |
      | 4        | tool_use | tool_result | permission denied: cannot remove /root        | 30          |
      | 5        | tool_use | tool_use    | Bash {"command":"rm -rf /root"}               | 50          |
    When I run the permission loop detector
    Then 1 permission loop pattern should be detected
    And the pattern should have tool_name "Bash"
    And the pattern should report 2 retries
    And the tokens_wasted should be 100
    And the tokens_wasted should exclude the initial legitimate attempt of 50 tokens

  Scenario: No loop when tool succeeds after one retry
    Given a session with the following spans:
      | position | phase    | block_type  | text                                          | token_count |
      | 1        | tool_use | tool_use    | Read {"file_path":"/etc/shadow"}              | 40          |
      | 2        | tool_use | tool_result | access denied                                 | 20          |
      | 3        | tool_use | tool_use    | Read {"file_path":"/etc/shadow"}              | 40          |
    When I run the permission loop detector
    Then 0 permission loop patterns should be detected
    # Because only 1 retry is below the min_retries threshold of 2

  Scenario: All denial keyword variants are recognised
    Given a session with the following spans:
      | position | phase    | block_type  | text                                          | token_count |
      | 1        | tool_use | tool_use    | Write {"file_path":"/sys/conf"}               | 60          |
      | 2        | tool_use | tool_result | Error: not allowed to write here              | 25          |
      | 3        | tool_use | tool_use    | Write {"file_path":"/sys/conf"}               | 60          |
      | 4        | tool_use | tool_result | EACCES: permission check failed               | 25          |
      | 5        | tool_use | tool_use    | Write {"file_path":"/sys/conf"}               | 60          |
      | 6        | tool_use | tool_result | unauthorized access attempt                   | 25          |
      | 7        | tool_use | tool_use    | Write {"file_path":"/sys/conf"}               | 60          |
    When I run the permission loop detector
    Then 1 permission loop pattern should be detected
    And the pattern should report 3 retries
    And the keywords "not allowed", "eacces", and "unauthorized" should all trigger denial detection

  Scenario: Different tool names break the permission loop chain
    Given a session with the following spans:
      | position | phase    | block_type  | text                                          | token_count |
      | 1        | tool_use | tool_use    | Bash {"command":"cat /root/secret"}           | 50          |
      | 2        | tool_use | tool_result | permission denied                             | 20          |
      | 3        | tool_use | tool_use    | Read {"file_path":"/root/secret"}             | 50          |
      | 4        | tool_use | tool_result | permission denied                             | 20          |
      | 5        | tool_use | tool_use    | Bash {"command":"cat /root/secret"}           | 50          |
    When I run the permission loop detector
    Then 0 permission loop patterns should be detected
    # Because the tool name changed from "Bash" to "Read" breaking the chain
