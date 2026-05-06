Feature: Task Complexity Estimation
  As a developer selecting model tiers
  I want automatic complexity estimation from intent text
  So that I use the right model and thinking budget for each task

  Scenario: Simple task classified as SIMPLE
    Given an intent text "fix the typo in README.md"
    When complexity is estimated
    Then the complexity tier is SIMPLE
    And confidence is above 0.5

  Scenario: Standard task classified as STANDARD
    Given an intent text "implement a new API endpoint for user authentication with JWT tokens and add error handling for invalid credentials"
    When complexity is estimated
    Then the complexity tier is STANDARD

  Scenario: Complex task classified as COMPLEX
    Given an intent text "refactor the entire authentication system across multiple files to use OAuth2"
    When complexity is estimated
    Then the complexity tier is COMPLEX

  Scenario: Multi-file cues increase complexity score
    Given an intent text mentioning "across multiple files" or "refactor the codebase"
    When complexity is estimated
    Then the complexity score includes a multi-file cue contribution with weight 3.0

  Scenario: Architecture cues increase complexity score
    Given an intent text mentioning "architecture" or "system design"
    When complexity is estimated
    Then the complexity score includes an architecture cue contribution with weight 3.0
