Feature: Session Health Report
  As a developer preparing for analysis
  I want a pre-analysis health check
  So that I know estimated time and content distribution before committing

  Scenario: Health report estimates total tokens
    Given a parsed session with multiple content blocks
    When a health report is generated
    Then estimated_total_tokens is a positive integer

  Scenario: Health report shows content distribution
    Given a parsed session with reasoning, tool_use, and generation blocks
    When a health report is generated
    Then the content_distribution includes counts for text, thinking, tool_use, and tool_result

  Scenario: Health report estimates analysis time
    Given a parsed session with approximately 1000 spans
    When a health report is generated
    Then estimated_analysis_seconds is approximately 0.5 seconds

  Scenario: Health report counts user and assistant messages
    Given a parsed session with 5 user messages and 8 assistant messages
    When a health report is generated
    Then user_count is 5 and assistant_count is 8
