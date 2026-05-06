Feature: Session Tagging
  As a developer categorising sessions
  I want to tag sessions and query stats by tag
  So that I can compare efficiency across categories like "refactor" or "bugfix"

  Background:
    Given a temporary TER history file with recorded sessions

  Scenario: Add tags to a session
    Given a recorded session with session_id "session-1"
    When tag_session is called with tags "refactor" and "auth"
    Then session "session-1" has tags "refactor" and "auth"

  Scenario: Tags are deduplicated
    Given a session already tagged with "refactor"
    When tag_session is called with tags "refactor" and "bugfix"
    Then the session has tags "refactor" and "bugfix" without duplicates

  Scenario: Stats by tag returns aggregate metrics
    Given 3 sessions tagged "refactor" with TERs 0.70, 0.80, and 0.75
    When get_stats_by_tag is called for "refactor"
    Then the result includes session_count of 3
    And the average TER is approximately 0.75
