Feature: TER History and Trending
  As a developer tracking efficiency over time
  I want TER history stored and trend analysis computed
  So that I can see whether my sessions are improving

  Background:
    Given a temporary TER history file

  Scenario: Record a TER result to history
    Given a TER result with session_id "session-1" and aggregate_ter 0.75
    When the result is recorded to history
    Then the history file contains an entry for "session-1"

  Scenario: Retrieve trend for a project
    Given 5 recorded TER results for project "/app"
    When get_trend is called for project "/app"
    Then a list of 5 TERHistoryEntry objects is returned

  Scenario: Trend direction computed from first and second half averages
    Given 10 recorded results where the first 5 average TER 0.60 and the last 5 average 0.75
    When get_summary is called
    Then trend_direction is "improving"

  Scenario: Stable trend when difference is below 0.02
    Given 10 recorded results where the first 5 average TER 0.70 and the last 5 average 0.71
    When get_summary is called
    Then trend_direction is "stable"

  Scenario: Declining trend when TER drops
    Given 10 recorded results where the first 5 average TER 0.80 and the last 5 average 0.60
    When get_summary is called
    Then trend_direction is "declining"

  Scenario: Summary includes best and worst TER
    Given recorded results with TERs 0.45, 0.67, 0.82, 0.91
    When get_summary is called
    Then best_ter is 0.91 and worst_ter is 0.45
