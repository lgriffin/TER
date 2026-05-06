Feature: Embedding Cache and Span Optimisation
  As a developer processing large sessions
  I want span embeddings cached and spans merged
  So that re-analysis is near-instant and embedding count is reduced

  Scenario: Merge adjacent same-phase spans
    Given 5 consecutive reasoning phase spans
    When merge_adjacent_spans is called
    Then 1 merged span is produced
    And the merged span text concatenates all 5 span texts

  Scenario: Different phases prevent merging
    Given a reasoning span followed by a tool_use span followed by a reasoning span
    When merge_adjacent_spans is called
    Then 3 separate merged spans are produced

  Scenario: Filter short spans below minimum token count
    Given spans with token counts 5, 50, 8, 100, and 3
    And the minimum token count is 10
    When filter_short_spans is called
    Then 2 spans are embeddable with 50 and 100 tokens
    And 3 spans are skipped with default_confidence of 0.1

  Scenario: GPU detection falls back to CPU
    When detect_device is called without GPU hardware
    Then the device is "cpu"
    And batch_size_hint is 64

  Scenario: Batch embedding produces correct dimensions
    Given a list of 10 text strings
    When compute_batch_embeddings is called
    Then 10 embedding vectors are returned
    And each has 384 dimensions
