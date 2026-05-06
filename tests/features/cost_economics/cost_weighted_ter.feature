Feature: Cost-Weighted TER Computation
  As a developer tracking session costs
  I want TER weighted by dollar cost instead of raw token count
  So that I understand the financial impact of waste

  Background:
    Given the default pricing tier is "sonnet"

  Scenario: Cost-weighted TER penalises waste output tokens more than input
    Given a session with 1000 waste output tokens and 1000 waste input tokens
    When cost-weighted TER is computed for the "sonnet" tier
    Then the waste cost from output tokens exceeds the waste cost from input tokens

  Scenario: Compute total session cost in USD
    Given a session with classified spans totalling 10000 tokens
    When cost-weighted TER is computed
    Then total_cost_usd is a positive dollar amount
    And aligned_cost_usd plus waste_cost_usd equals total_cost_usd

  Scenario: Cost-weighted TER differs from raw TER
    Given a session where waste is concentrated in expensive output tokens
    When cost-weighted TER is computed
    Then cost_weighted_ter is lower than raw_ter

  Scenario: Cached tokens are costed at reduced rate
    Given a session with 5000 cached read tokens on the "sonnet" tier
    When cost-weighted TER is computed
    Then cached read tokens are billed at $0.30 per million not $3.00 per million

  Scenario: Thinking tokens billed at output rate
    Given a session with reasoning phase spans
    When cost-weighted TER is computed
    Then reasoning tokens are categorised as THINKING
    And THINKING tokens are billed at the output rate
