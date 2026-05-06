Feature: Session-Level Validation
  As a developer preparing to analyse a session
  I want cross-message validation checks
  So that I detect structural problems before analysis

  Scenario: Valid session with user and assistant messages
    Given a parsed session with user and assistant messages
    When session validation runs
    Then the result is valid

  Scenario: Session with no user messages is flagged
    Given a parsed session with only assistant messages
    When session validation runs
    Then a warning about missing user messages is reported

  Scenario: Session with no assistant messages is flagged
    Given a parsed session with only user messages
    When session validation runs
    Then a warning about missing assistant messages is reported

  Scenario: Out-of-order timestamps generate a warning
    Given a parsed session where a later message has an earlier timestamp
    When session validation runs
    Then a warning about timestamp ordering is reported

  Scenario: Content block counts are accurate
    Given a parsed session with 5 text blocks, 3 tool_use blocks, and 2 thinking blocks
    When session validation runs
    Then the result reports content_block_count of 10
