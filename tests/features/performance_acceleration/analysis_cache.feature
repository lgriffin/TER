Feature: Analysis Cache
  As a developer re-analysing sessions with different thresholds
  I want intermediate results cached on disk
  So that expensive steps like embedding are skipped on re-analysis

  Background:
    Given a temporary cache directory
    And the default cache TTL is 168 hours

  Scenario: Cache miss triggers computation and stores result
    Given a cache key that has not been stored
    When get_or_compute is called with a compute function
    Then the compute function is invoked
    And the result is stored in the cache
    And cache_stats reports miss_count of 1

  Scenario: Cache hit returns stored result without recomputation
    Given a cache key with a previously stored result
    When get_or_compute is called with the same key
    Then the compute function is not invoked
    And the previously stored result is returned
    And cache_stats reports hit_count of 1

  Scenario: Expired cache entries are evicted
    Given a cache entry older than the TTL
    When get_or_compute is called for that key
    Then the entry is treated as a miss
    And the compute function is invoked

  Scenario: Invalidate clears entries for a specific session
    Given cached results for a session file
    When invalidate is called for that session path
    Then the cached entries for that session are removed

  Scenario: Clear all purges the entire cache
    Given a cache with multiple entries
    When clear_all is called
    Then cache_stats reports entry_count of 0
    And hit_count and miss_count are reset to 0
