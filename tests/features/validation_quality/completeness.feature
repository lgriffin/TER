Feature: Session Completeness Assessment
  As a developer analysing a session
  I want to know if the session ended cleanly
  So that I can account for incomplete data in TER interpretation

  Scenario: Complete session scores 1.0
    Given a session where the last assistant message has stop_reason "end_turn"
    And all tool_use blocks have matching tool_result responses
    When completeness is assessed
    Then is_complete is true
    And completeness_score is 1.0

  Scenario: Session ending mid-tool-use is incomplete
    Given a session where the last message is a tool_use with no tool_result
    When completeness is assessed
    Then is_complete is false
    And the issues list mentions unresolved tool calls

  Scenario: Session with unmatched tool_use reduces completeness
    Given a session with 3 tool_use blocks and only 2 tool_result blocks
    When completeness is assessed
    Then completeness_score is below 1.0
    And the issues list reports the unmatched tool_use

  Scenario: Session without end_turn stop_reason is flagged
    Given a session where the last assistant message has no stop_reason
    When completeness is assessed
    Then the issues list mentions missing stop_reason
