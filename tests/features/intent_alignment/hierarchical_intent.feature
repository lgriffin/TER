Feature: Hierarchical Intent Extraction
  As a developer with evolving session goals
  I want a high-level intent with sub-intents
  So that spans are scored against the most specific applicable goal

  Background:
    Given a HierarchicalIntentExtractor with sub_intent_weight 0.7

  Scenario: First prompt becomes the high-level intent
    Given user prompts:
      | prompt                          |
      | Build an authentication system  |
      | Add JWT token support           |
      | Add password reset              |
    When hierarchical intent extraction runs
    Then the first IntentVector represents the high-level intent
    And 2 additional sub-intent IntentVectors are returned

  Scenario: Span scoring blends high-level and sub-intent similarity
    Given a high-level intent about "authentication"
    And a sub-intent about "JWT tokens"
    And a span about "generating JWT tokens"
    When the span is scored against the intents
    Then the blended score uses 70 percent sub-intent and 30 percent high-level similarity

  Scenario: Single prompt returns only high-level intent
    Given user prompts:
      | prompt                     |
      | Fix the bug in auth.py     |
    When hierarchical intent extraction runs
    Then exactly 1 IntentVector is returned

  Scenario Outline: Confidence scales by prompt word count
    Given a prompt with <word_count> words
    When intent confidence is computed
    Then confidence is <expected_confidence>

    Examples:
      | word_count | expected_confidence |
      | 1          | 0.2                 |
      | 3          | 0.5                 |
      | 8          | 0.7                 |
      | 15         | 0.85                |
