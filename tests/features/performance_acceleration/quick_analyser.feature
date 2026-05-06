Feature: Quick Analyser
  As a developer wanting fast approximate TER
  I want a keyword-based analyser that skips embedding
  So that I get results in 1-2 seconds for any session size

  Background:
    Given a QuickAnalyser with top_n_keywords 30

  Scenario: Quick analysis produces TER-compatible result
    Given a valid session JSONL file
    When quick analysis is run
    Then the result contains session_id and aggregate_ter
    And the result method is "quick_keyword"
    And total_tokens equals aligned_tokens plus waste_tokens

  Scenario: Keywords extracted exclude stop words
    Given user prompts mentioning "login", "authentication", and "the"
    When keywords are extracted
    Then the keyword set includes "login" and "authentication"
    And the keyword set excludes "the"

  Scenario: Empty session returns TER of 1.0
    Given a session JSONL file with no content spans
    When quick analysis is run
    Then aggregate_ter is 1.0

  Scenario: No meaningful keywords treats all tokens as aligned
    Given a session with user prompts containing only stop words
    When quick analysis is run
    Then aggregate_ter is 1.0

  Scenario: Keyword overlap score is fraction of keywords found
    Given a set of 10 keywords
    And a span text containing 4 of those keywords
    When keyword overlap is computed
    Then the score is 0.4

  Scenario: Missing session file raises FileNotFoundError
    Given a path to a non-existent JSONL file
    When quick analysis is run
    Then a FileNotFoundError is raised

  Scenario: Deduplication keeps entry with highest output_tokens
    Given a session file with duplicate requestIds having different output_tokens
    When quick analysis parses the session
    Then only the entry with the highest output_tokens is kept per requestId
