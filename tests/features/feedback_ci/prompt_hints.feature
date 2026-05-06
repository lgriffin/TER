Feature: Actionable Prompt Hints
  As a developer who received a TER report
  I want improvement suggestions derived from waste patterns
  So that I know how to write better prompts

  Scenario: Low reasoning phase score generates a hint
    Given a TER result with reasoning phase score below 0.5
    When prompt hints are generated
    Then at least one hint has category related to reasoning

  Scenario: Waste patterns generate related hints
    Given a TER result with a "reasoning_loop" waste pattern
    When prompt hints are generated
    Then at least one hint references the reasoning loop pattern
    And the hint includes an estimated_impact

  Scenario: High TER generates no hints
    Given a TER result with aggregate_ter above 0.9 and no waste patterns
    When prompt hints are generated
    Then an empty list of hints is returned

  Scenario: Each hint has required fields
    Given a TER result with multiple waste patterns
    When prompt hints are generated
    Then each hint has category, suggestion, estimated_impact, and related_pattern_type
