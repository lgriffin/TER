Feature: Parallel Span Embedding
  As a developer processing sessions with many spans
  I want embeddings computed in parallel across CPU cores
  So that large sessions are processed faster

  Scenario: Small inputs use single-process fallback
    Given a list of 50 texts
    When parallel_embed is called
    Then single-process embedding is used
    And 50 embedding vectors are returned

  Scenario: Large inputs use multiprocessing
    Given a list of 150 texts
    When parallel_embed is called with default workers
    Then multiprocessing is used with at most 4 workers
    And 150 embedding vectors are returned in order

  Scenario: Worker count capped at 4
    Given a system with 8 CPU cores
    When parallel_embed determines worker count
    Then n_workers is 4

  Scenario: Multiprocessing failure falls back to single process
    Given parallel embedding encounters a multiprocessing error
    When parallel_embed runs
    Then it falls back to single-process embedding
    And returns correct results
