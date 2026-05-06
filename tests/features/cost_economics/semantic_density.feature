Feature: Semantic Density Scoring
  As a developer evaluating output quality
  I want to measure information density per token
  So that I can identify verbose or repetitive output

  Scenario: High-density text scores above 0.5
    Given a text with diverse vocabulary and no repetition
    When semantic density is computed
    Then density_score is above 0.5
    And vocabulary_richness is above 0.5
    And redundancy_ratio is below 0.1

  Scenario: Highly repetitive text has high redundancy
    Given a text repeating the same sentence 5 times
    When semantic density is computed
    Then redundancy_ratio is above 0.3
    And density_score is below 0.5

  Scenario: Empty text has zero density
    Given an empty text string
    When semantic density is computed
    Then density_score is 0.0

  Scenario: Density score formula uses correct weights
    Given a text with known vocabulary_richness, information_entropy, and redundancy
    When semantic density is computed
    Then density_score equals 0.4 times vocabulary_richness plus 0.4 times normalised_entropy plus 0.2 times one minus redundancy_ratio
