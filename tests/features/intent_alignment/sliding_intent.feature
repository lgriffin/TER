Feature: Sliding Window Intent Extraction
  As a developer with multi-turn sessions
  I want intent segmented by topic shifts
  So that spans are scored against the nearest relevant intent

  Background:
    Given a SlidingIntentExtractor with window_size 5 and split_threshold 0.45

  Scenario: Single prompt produces one IntentVector
    Given user prompts:
      | prompt             |
      | Add a login page   |
    When sliding intent extraction runs
    Then exactly 1 IntentVector is returned
    And the IntentVector embedding has 384 dimensions

  Scenario: Diverging prompts create multiple segments
    Given user prompts that shift topic from "login" to "database migration"
    And the cosine similarity between adjacent prompts drops below 0.45
    When sliding intent extraction runs
    Then 2 or more IntentVector objects are returned

  Scenario: Similar prompts stay in the same segment
    Given 3 user prompts all about authentication
    And their pairwise cosine similarity is above 0.45
    When sliding intent extraction runs
    Then exactly 1 IntentVector is returned

  Scenario: Window size enforces segment splits
    Given 7 prompts all on the same topic
    When sliding intent extraction runs
    Then at least 2 segments are produced

  Scenario: Empty prompts produce a default intent
    Given no user prompts
    When sliding intent extraction runs
    Then 1 IntentVector is returned with empty text and confidence 0.0
