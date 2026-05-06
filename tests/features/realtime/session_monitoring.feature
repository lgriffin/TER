Feature: Session File Monitoring
  As a developer running a live TER monitor
  I want the SessionMonitor to poll a JSONL session file for new content
  So that TER signals are emitted as the session progresses

  Background:
    Given a temporary directory with a JSONL session file
    And a SessionMonitor configured with a poll interval of 2.0 seconds

  Scenario: Poll detects new lines appended to the session file
    Given the session file contains 3 JSONL lines with assistant messages
    When poll_once is called
    Then TERSignal objects are returned for the new assistant messages
    When 2 more JSONL lines with assistant messages are appended to the file
    And poll_once is called again
    Then only the 2 new lines produce TERSignal objects
    And previously processed lines are not re-processed

  Scenario: current_ter returns the aggregate TER as a valid float
    Given the session file contains assistant messages with known token counts
    When poll_once is called
    Then the current_ter property returns a float between 0.0 and 1.0

  Scenario: Non-existent file is handled gracefully without errors
    Given a SessionMonitor pointing to a file that does not exist
    When poll_once is called
    Then an empty list of signals is returned
    And no exception is raised

  Scenario: Callback is invoked once per emitted signal
    Given an on_signal callback is registered with the SessionMonitor
    And the session file contains 3 assistant messages
    When poll_once is called
    Then the callback is invoked exactly 3 times
    And each invocation receives a TERSignal object

  Scenario: stop() terminates the blocking poll loop
    Given the SessionMonitor is running its blocking poll loop in a thread
    When stop() is called from another thread
    Then the poll loop exits
    And the monitor's _stop flag is true
