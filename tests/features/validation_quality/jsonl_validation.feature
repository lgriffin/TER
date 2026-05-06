Feature: JSONL Line and File Validation
  As a developer loading session data
  I want each JSONL line validated against the expected schema
  So that I catch data issues before expensive analysis

  Scenario: Valid message line passes validation
    Given a JSONL line with type, uuid, sessionId, and message containing role and content
    When the line is validated
    Then the result is valid with no errors

  Scenario: Invalid JSON fails validation
    Given a JSONL line containing malformed JSON
    When the line is validated
    Then the result is invalid
    And the error message includes "Invalid JSON"

  Scenario: Missing required top-level fields reported
    Given a JSONL line missing the "message" field
    When the line is validated
    Then the error reports missing required fields

  Scenario Outline: Known meta line types are accepted
    Given a JSONL line with type "<meta_type>"
    When the line is validated
    Then the result is valid

    Examples:
      | meta_type               |
      | attachment              |
      | file-history-snapshot   |
      | last-prompt             |
      | permission-mode         |
      | progress                |
      | queue-operation         |
      | summary                 |
      | system                  |

  Scenario: Unknown content block type generates a warning
    Given a content block with type "custom_block"
    When the line is validated
    Then a warning about unknown block type is reported
    But the line is still valid

  Scenario: tool_use block missing name field is invalid
    Given a content block of type "tool_use" without a "name" field
    When the line is validated
    Then an error is reported about missing name

  Scenario: Negative token usage is invalid
    Given a message with negative output_tokens
    When the line is validated
    Then an error is reported about non-negative token counts

  Scenario: Validate entire JSONL file
    Given a JSONL file with 100 lines and 3 malformed lines
    When validate_jsonl_file is called
    Then the result reports total_lines of 100 and valid_lines of 97
    And error_lines contains the 3 malformed line numbers

  Scenario: Non-existent file raises FileNotFoundError
    Given a path to a non-existent JSONL file
    When validate_jsonl_file is called
    Then a FileNotFoundError is raised
