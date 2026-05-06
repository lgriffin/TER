Feature: Model Pricing Tiers
  As a developer choosing a model
  I want accurate pricing per tier
  So that cost calculations reflect actual API rates

  Scenario: Haiku pricing is correctly configured
    Given the "haiku" pricing tier
    Then input_per_mtok is 0.80
    And output_per_mtok is 4.00
    And cached_read_per_mtok is 0.08
    And cached_write_per_mtok is 1.00

  Scenario: Sonnet pricing is correctly configured
    Given the "sonnet" pricing tier
    Then input_per_mtok is 3.00
    And output_per_mtok is 15.00
    And cached_read_per_mtok is 0.30
    And cached_write_per_mtok is 3.75

  Scenario: Opus pricing is correctly configured
    Given the "opus" pricing tier
    Then input_per_mtok is 15.00
    And output_per_mtok is 75.00
    And cached_read_per_mtok is 1.50
    And cached_write_per_mtok is 18.75

  Scenario Outline: Different tiers produce different output costs
    Given a session with 5000 output tokens
    When cost is computed for tier "<tier>"
    Then the output cost uses rate <rate> per million tokens

    Examples:
      | tier   | rate  |
      | haiku  | 4.00  |
      | sonnet | 15.00 |
      | opus   | 75.00 |
