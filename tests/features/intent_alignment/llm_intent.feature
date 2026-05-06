Feature: LLM-Assisted Intent Extraction
  As a developer with complex or ambiguous prompts
  I want Claude to summarise my intent as a structured goal
  So that the intent embedding is more accurate than direct embedding

  Scenario: LLM produces a StructuredGoal
    Given an LLM intent extractor with a valid API key
    And user prompts:
      | prompt                                      |
      | Build a REST API for user management        |
      | Support pagination and filtering            |
    When LLM intent extraction runs
    Then a StructuredGoal is produced with primary_goal, sub_goals, constraints, and expected_outputs
    And the IntentVector confidence is 0.95

  Scenario: Fallback to direct embedding when no API key
    Given an LLM intent extractor with no API key
    And user prompts:
      | prompt              |
      | Build a REST API    |
    When LLM intent extraction runs
    Then the fallback produces an IntentVector by direct embedding
    And no error is raised

  Scenario: StructuredGoal flattens to embedding text
    Given a StructuredGoal with primary_goal "Build API"
    And sub_goals "Add users endpoint" and "Add pagination"
    And constraints "Use Python"
    And expected_outputs "api.py"
    When to_embedding_text is called
    Then the output combines all fields separated by pipes

  Scenario: Factory creates correct strategy by name
    When create_intent_extractor is called with strategy "sliding"
    Then a SlidingIntentExtractor is returned
    When create_intent_extractor is called with strategy "hierarchical"
    Then a HierarchicalIntentExtractor is returned
    When create_intent_extractor is called with strategy "llm"
    Then a LLMIntentExtractor is returned

  Scenario: Factory rejects unknown strategy
    When create_intent_extractor is called with strategy "unknown"
    Then a ValueError is raised
