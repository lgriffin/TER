Feature: CI Threshold Checking
  As a CI pipeline operator
  I want to gate deployments on TER quality
  So that sessions below a minimum efficiency are flagged

  Scenario: Session above threshold passes
    Given a TER result with aggregate_ter 0.80
    When check_threshold is called with threshold 0.70
    Then the check passes
    And the result message indicates the TER exceeds the threshold

  Scenario: Session below threshold fails
    Given a TER result with aggregate_ter 0.55
    When check_threshold is called with threshold 0.70
    Then the check fails
    And the result message indicates the TER is below the threshold

  Scenario: Per-phase threshold checking
    Given a TER result with reasoning score 0.90, tool_use score 0.40, and generation score 0.80
    When check_threshold is called with aggregate threshold 0.60 and phase threshold 0.50
    Then the check fails because tool_use score 0.40 is below the phase threshold 0.50
    And phase_failures includes "tool_use"

  Scenario: Threshold at exact boundary passes
    Given a TER result with aggregate_ter 0.70
    When check_threshold is called with threshold 0.70
    Then the check passes
