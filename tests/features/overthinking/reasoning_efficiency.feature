Feature: Reasoning Efficiency and Overthinking Detection
  As a developer optimising thinking token budgets
  I want to detect when reasoning stops adding value
  So that I can recommend optimal thinking token budgets

  Scenario: Detect overthinking when novelty plateaus
    Given a session with 10 reasoning segments
    And novelty scores decline steadily after segment 6
    When analyze_overthinking is called
    Then is_overthinking is true
    And optimal_cutoff_index is approximately 6
    And wasted_reasoning_tokens covers segments after the cutoff

  Scenario: No overthinking when novelty remains high
    Given a session with 5 reasoning segments
    And each segment introduces significant new information
    When analyze_overthinking is called
    Then is_overthinking is false
    And reasoning_efficiency is above 0.8

  Scenario: Classify reasoning phases
    Given a reasoning segment containing "wait" and "actually" and "therefore"
    When the segment is classified
    Then it is classified as a high-value segment with elevated high_value_token_count

  Scenario: Detect filler patterns
    Given a reasoning segment containing "let me re-read" and "let me check again"
    When the segment is classified
    Then filler_ratio is above 0.0

  Scenario: Optimal cutoff uses novelty threshold of 0.15
    Given reasoning segments with novelty scores 0.8, 0.6, 0.4, 0.2, 0.1, 0.05
    When find_optimal_cutoff is called with novelty_threshold 0.15
    Then the cutoff index is 4

  Scenario: Recommended budget based on useful tokens
    Given an overthinking result with 10000 total reasoning tokens and 6000 useful tokens
    When the recommended budget is computed
    Then recommended_budget is approximately 6000 tokens
