Feature: Live Dashboard Multi-Session Monitoring
  As a developer overseeing multiple Claude Code sessions
  I want the LiveDashboard to discover and monitor all active sessions in a project
  So that I can see aggregated efficiency metrics across the project

  Background:
    Given a temporary project directory

  Scenario: Dashboard discovers JSONL session files via recursive glob
    Given the project directory contains the following JSONL files:
      | path                          |
      | sessions/session_a.jsonl      |
      | sessions/deep/session_b.jsonl |
    When a LiveDashboard is created for the project directory
    And poll_once is called
    Then 2 SessionMonitor instances are created
    And each monitor corresponds to one of the discovered JSONL files

  Scenario: New session files are detected on subsequent polls
    Given the project directory contains 1 JSONL session file
    When poll_once is called
    Then 1 SessionMonitor instance exists
    When a new file "sessions/session_new.jsonl" is created in the project directory
    And poll_once is called again
    Then 2 SessionMonitor instances exist
    And the new session is tracked without affecting the existing monitor

  Scenario: get_summary returns correct aggregate metrics
    Given the project directory contains 2 JSONL session files
    And session A has aggregate TER 0.80 with 1000 total tokens and 200 waste tokens
    And session B has aggregate TER 0.60 with 2000 total tokens and 800 waste tokens
    When get_summary is called
    Then the summary session_count is 2
    And the summary average_ter is 0.70
    And the summary total_tokens is 3000
    And the summary total_waste is 1000

  Scenario: Empty project directory produces no monitors
    Given the project directory contains no JSONL files
    When a LiveDashboard is created for the project directory
    And poll_once is called
    Then 0 SessionMonitor instances are created
    And get_summary returns session_count 0 and average_ter 0.0
