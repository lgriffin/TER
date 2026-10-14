Feature: L4 Advisory policies
  As a developer running TER's live advice
  I want advice only from policies I approved, on evidence I can check
  So that TER never nudges an agent session on a guess

  The Intervention Engine reads structured detector signals and answers each
  one under each approved policy that names its detector. Every scenario cites
  the EARS requirement it verifies with a req tag, so it counts in the trace
  gate like any other test (docs/guides/testing.md).

  Background:
    Given a policy "reuse-reads" that warns "Reuse the earlier read." on repeated_exploration
    And the policy stays silent for 5 tool events after it advises
    And calibration marks repeated_exploration eligible

  @req:TER-INT-015
  Scenario: A policy file nobody approved gives no advice
    Given the policy file is not approved
    When TER builds the intervention engine
    Then TER refuses with "policy approve"

  @req:TER-INT-015
  Scenario: Editing an approved policy file voids its approval
    Given the policy file is approved by "Leigh"
    When someone edits the advice to "Something else."
    Then the policy file's approval state is "changed since approval"
    And building the intervention engine is refused with "changed after Leigh"

  @req:TER-INT-003 @req:TER-INT-014
  Scenario: An approved policy advises with the signal and its evidence
    Given the policy file is approved by "Leigh"
    When a repeated_exploration signal with confidence 0.85 cites events "e1, e2"
    Then TER advises "Reuse the earlier read."
    And the advice shows the repeated_exploration signal and events "e1, e2"

  @req:TER-INT-007
  Scenario: A signal that cites no event is never advised
    Given the policy file is approved by "Leigh"
    When a repeated_exploration signal with confidence 0.85 cites no events
    Then the outcome is "no_evidence"

  @req:TER-INT-010
  Scenario: A detector calibration has not marked eligible is never advised
    Given calibration marks no detector eligible
    And the policy file is approved by "Leigh"
    When a repeated_exploration signal with confidence 0.85 cites events "e1, e2"
    Then the outcome is "ineligible"

  @req:TER-INT-010
  Scenario Outline: An uncertain signal counts as waste but never triggers advice
    Given the policy file is approved by "Leigh"
    When a repeated_exploration signal with confidence <confidence> cites events "e1, e2"
    Then the outcome is "<outcome>"

    Examples:
      | confidence | outcome        |
      | 0.69       | low_confidence |
      | 0.70       | advised        |

  @req:TER-INT-016
  Scenario: A policy is silent during its cooldown
    Given the policy file is approved by "Leigh"
    When repeated_exploration signals arrive at tool events 1, 3 and 6
    Then the outcomes are "advised, cooldown, advised"

  @req:TER-INT-002
  Scenario: Below the L4 ceiling an approved policy still advises nothing
    Given the policy file is approved by "Leigh"
    And the maturity ceiling is L3
    When a repeated_exploration signal with confidence 0.85 cites events "e1, e2"
    Then the outcome is "below_ceiling"
