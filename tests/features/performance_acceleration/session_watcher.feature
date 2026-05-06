Feature: Session Watcher
  As a developer
  I want automatic detection of new and modified session files
  So that analysis runs without manual invocation

  Background:
    Given the default watch polling interval is 30 seconds

  Scenario: Detect new session file
    Given a watched directory with 2 existing JSONL files
    When a new JSONL file is added to the directory
    And the watcher polls
    Then a NEW_SESSION event is emitted for the new file

  Scenario: Detect modified session file
    Given a watched directory with an existing JSONL file
    When the file modification time changes
    And the watcher polls
    Then a MODIFIED_SESSION event is emitted

  Scenario: Callback invoked on detection
    Given a watcher with a registered callback
    When a new session file appears and the watcher polls
    Then the callback receives a WatchEvent with the file path and timestamp

  Scenario: Stop terminates the watch loop
    Given a running watcher
    When stop is called
    Then the watcher exits its polling loop
