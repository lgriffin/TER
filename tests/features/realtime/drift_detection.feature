Feature: TER Drift Detection
  As a developer monitoring session efficiency trends
  I want drift detection to classify TER trajectory over a rolling window
  So that I receive timely warnings when efficiency is degrading

  Background:
    Given the drift window size is 5
    And the drift threshold is 0.10

  Scenario: Stable TER values produce STABLE direction
    When the recent TER values are [0.80, 0.81, 0.80, 0.79, 0.80]
    Then the drift direction is STABLE
    And the drift magnitude is below 0.10

  Scenario: Declining TER values produce DEGRADING direction
    When the recent TER values are [0.90, 0.80, 0.70, 0.60, 0.50]
    Then the drift direction is DEGRADING
    And the drift magnitude is at least 0.10
    And the magnitude equals abs(slope * 5)

  Scenario: Improving TER values produce IMPROVING direction
    When the recent TER values are [0.50, 0.60, 0.70, 0.80, 0.90]
    Then the drift direction is IMPROVING
    And the drift magnitude is at least 0.10
    And the magnitude equals abs(slope * 5)

  Scenario: Fewer than 2 values produce STABLE with zero magnitude
    When the recent TER values are [0.75]
    Then the drift direction is STABLE
    And the drift magnitude is exactly 0.0

  Scenario: CAUTION warning emitted when degrading drift exceeds threshold
    Given a RollingTERState with recent TER values [0.85, 0.75, 0.65, 0.55, 0.45]
    When the next assistant message produces a TER that continues the decline
    Then the drift direction is DEGRADING
    And the drift magnitude exceeds 0.10
    And the TERSignal warning_level is CAUTION
    And the warnings list contains a message matching "TER dropped .* over last 5 messages"

  Scenario: ALERT warning when current TER falls below 0.4
    Given a RollingTERState where aggregate TER is 0.35
    When a TERSignal is emitted
    Then the TERSignal warning_level is ALERT
    And the warnings list contains a message matching "TER is critically low.*session may be spiralling"

  Scenario: is_healthy property reflects INFO level and non-DEGRADING drift
    Given a TERSignal with warning_level INFO and drift direction STABLE
    Then the signal is_healthy is true
    Given a TERSignal with warning_level INFO and drift direction DEGRADING
    Then the signal is_healthy is false
    Given a TERSignal with warning_level CAUTION and drift direction STABLE
    Then the signal is_healthy is false

  Scenario: Waste warning when total tokens exceed 5000 with low alignment ratio
    Given a RollingTERState with total_tokens 6000 and aligned_tokens 2500
    When a TERSignal is emitted
    Then the raw_ratio is below 0.5
    And the warnings list contains a message matching "Over half of tokens.*classified as waste"
    And the warning_level is at least CAUTION
