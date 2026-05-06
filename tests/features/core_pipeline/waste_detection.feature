Feature: Waste Pattern Detection
  As a developer reviewing a Claude Code session
  I want to see specific waste patterns identified in the session
  So that I understand where token waste occurred and can take corrective action

  The waste detector identifies three pattern types:
  - Reasoning loops: 3 or more consecutive redundant reasoning spans
  - Duplicate tool calls: identical tool name and parameters within a 5-step window
  - Context restatement: response spans with cosine similarity above 0.85 to prior responses

  Background:
    Given the reasoning loop threshold is 3 consecutive redundant spans
    And the duplicate tool call window is 5 steps
    And the context restatement similarity threshold is 0.85

  # ── Scenario 1: Reasoning loop detection ─────────────────────────────────

  Scenario: Detect reasoning loop with 3 or more consecutive redundant spans
    Given a session containing the following reasoning spans:
      | position | text                                                    |
      | 1        | I need to create a login page with email and password.  |
      | 2        | I should create a login page with email and password.   |
      | 3        | Let me create a login page with email and password.     |
      | 4        | Creating a login page with email and password fields.   |
    And spans at positions 2, 3, and 4 are redundant with span at position 1
    When waste patterns are analyzed
    Then a "reasoning_loop" pattern should be reported
    And the pattern should involve 3 redundant spans
    And the pattern should report the tokens_wasted consumed by the redundant spans

  # ── Scenario 2: Duplicate tool calls within window ───────────────────────

  Scenario: Detect duplicate tool calls within 5-step window
    Given a session containing the following tool calls:
      | position | tool_name | parameters                       |
      | 5        | Bash      | {"command": "ls -la /app"}       |
      | 7        | Read      | {"file_path": "/app/main.py"}    |
      | 8        | Bash      | {"command": "ls -la /app"}       |
    And the calls at positions 5 and 8 have identical name and parameters
    And positions 5 and 8 are within the 5-step window
    When waste patterns are analyzed
    Then a "duplicate_tool_call" pattern should be reported
    And the pattern details should identify the duplicated tool as "Bash"
    And the pattern should report the tokens_wasted for the duplicate call

  # ── Scenario 3: Context restatement detection ────────────────────────────

  Scenario: Detect context restatement when similarity exceeds 0.85
    Given a session containing the following response spans:
      | position | text                                                                    |
      | 10       | I have created the login page with email and password fields as requested. |
      | 15       | As I mentioned, I created the login page with email and password fields.   |
    And the cosine similarity between the spans at positions 10 and 15 is above 0.85
    When waste patterns are analyzed
    Then a "context_restatement" pattern should be reported
    And the pattern should report the tokens_wasted for the restated content

  # ── Scenario 4: No waste patterns found ──────────────────────────────────

  Scenario: Clean session reports no waste patterns
    Given a session where all reasoning spans introduce new information
    And all tool calls have unique name-parameter combinations
    And no response spans have cosine similarity above 0.85
    When waste patterns are analyzed
    Then the waste pattern report should indicate no patterns were found
    And the waste patterns list should be empty

  # ── Scenario 5: Below reasoning loop threshold ──────────────────────────

  Scenario: Exactly 2 redundant reasoning spans are not flagged as a loop
    Given a session containing the following reasoning spans:
      | position | text                                                    |
      | 1        | I need to create a login page with email and password.  |
      | 2        | I should create a login page with email and password.   |
      | 3        | Now let me check the project structure before coding.   |
    And only spans at positions 1 and 2 are redundant
    When waste patterns are analyzed
    Then no "reasoning_loop" pattern should be reported

  # ── Scenario 6: Duplicate calls outside window ──────────────────────────

  Scenario: Duplicate tool calls outside the 5-step window are not flagged
    Given a session containing the following tool calls:
      | position | tool_name | parameters                       |
      | 1        | Bash      | {"command": "ls -la /app"}       |
      | 10       | Bash      | {"command": "ls -la /app"}       |
    And the calls at positions 1 and 10 have identical name and parameters
    But positions 1 and 10 are outside the 5-step window
    When waste patterns are analyzed
    Then no "duplicate_tool_call" pattern should be reported

  # ── Scenario 7: Token counts are accurate ────────────────────────────────

  Scenario: Waste pattern reports accurate token counts
    Given a session containing a reasoning loop with the following spans:
      | position | token_count |
      | 3        | 120         |
      | 4        | 115         |
      | 5        | 130         |
    And spans at positions 3, 4, and 5 form a reasoning loop
    And the first span at position 3 is the original reasoning
    When waste patterns are analyzed
    Then a "reasoning_loop" pattern should be reported
    And the tokens_wasted should equal the sum of tokens in the redundant spans
    And the tokens_wasted should equal 245
