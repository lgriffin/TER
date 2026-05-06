Feature: Intent Extraction from Session Prompts
  As a developer running the TER calculator
  I want user intent to be automatically extracted from session prompts
  So that token spans can be evaluated for alignment against what I was trying to accomplish

  The intent extractor produces an IntentVector with:
  - text: combined user prompt text
  - embedding: 384-dimensional vector representation
  - confidence: float between 0.0 and 1.0
  - source_prompts: list of individual user prompts that formed the intent

  Background:
    Given the similarity threshold is 0.40
    And the embedding model produces 384-dimensional vectors

  # ── Scenario 1: Clear single prompt ──────────────────────────────────────

  Scenario: Clear single prompt produces a valid IntentVector
    Given a session with the user prompt "Add a login page with email and password"
    When intent is extracted
    Then the result should be a valid IntentVector
    And the intent text should contain "login page"
    And the intent confidence should be greater than 0.5
    And the source_prompts should contain exactly 1 prompt
    And the intent embedding should not be empty

  # ── Scenario 2: Multiple prompts combined ────────────────────────────────

  Scenario: Multiple user prompts are combined into a coherent intent
    Given a session with the following user prompts:
      | prompt                                         |
      | Add a login page with email and password        |
      | Also add form validation for the email field    |
      | Make sure the password has a minimum of 8 chars |
    When intent is extracted
    Then the result should be a valid IntentVector
    And the intent text should reflect all three prompts
    And the source_prompts should contain exactly 3 prompts
    And the intent confidence should be greater than 0.5

  # ── Scenario 3: Short or ambiguous prompt ────────────────────────────────

  Scenario: Short or ambiguous prompt produces low confidence intent
    Given a session with the user prompt "fix it"
    When intent is extracted
    Then the result should be a valid IntentVector
    And the intent confidence should be less than 0.5
    And the source_prompts should contain exactly 1 prompt

  # ── Scenario 4: Related spans score higher ──────────────────────────────

  Scenario: Spans related to the intent score higher similarity than unrelated spans
    Given a session with the user prompt "Add a login page with email and password"
    And a token span with text "Creating the login form with email input field"
    And a token span with text "Let me check the weather forecast for tomorrow"
    When intent is extracted
    And similarity is computed between the intent and both spans
    Then the related span should have higher cosine similarity than the unrelated span
    And the related span similarity should be above the threshold of 0.40
    And the unrelated span similarity should be below the threshold of 0.40

  # ── Scenario 5: No user prompts ─────────────────────────────────────────

  Scenario: Session with no user prompts produces empty intent
    Given a session with no user prompts
    When intent is extracted
    Then the intent text should be empty
    And the intent confidence should be 0.0
    And the source_prompts should be empty

  # ── Scenario 6: Embedding dimensionality ─────────────────────────────────

  Scenario: Intent embedding has exactly 384 dimensions
    Given a session with the user prompt "Refactor the authentication module"
    When intent is extracted
    Then the intent embedding should have exactly 384 dimensions
    And each dimension should be a numeric value
