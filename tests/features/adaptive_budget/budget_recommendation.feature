Feature: Token Budget Recommendation
  As a developer managing token spend
  I want budget recommendations based on task complexity
  So that I allocate thinking tokens efficiently

  Scenario Outline: Budget tiers map to thinking token limits
    Given an intent text classified as "<tier>"
    When a budget is recommended
    Then max_thinking_tokens is <budget>
    And model_tier is "<model>"

    Examples:
      | tier     | budget | model  |
      | SIMPLE   | 2048   | HAIKU  |
      | STANDARD | 8192   | SONNET |
      | COMPLEX  | 32768  | OPUS   |

  Scenario: Budget includes estimated total tokens and cost
    Given an intent text "add user authentication"
    When a budget is recommended
    Then estimated_total_tokens is a positive integer
    And estimated_cost_usd is a positive float
    And confidence is between 0.0 and 1.0

  Scenario: Historical adjustment modifies budget
    Given a HistoricalBudgetAnalyzer with past outcomes
    And past STANDARD tasks used an average of 12000 thinking tokens
    When a budget is recommended for a STANDARD task with history
    Then the budget is adjusted based on historical performance

  Scenario: Budget recommendation includes reasoning
    Given an intent text "implement dark mode"
    When a budget is recommended
    Then the reasoning field explains why the tier was chosen
