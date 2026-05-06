# Phase 2B: Dynamic Token Budget Controller via MCP

**Status:** Theoretical Architecture (Pre-Implementation)  
**Date:** 2026-05-05  
**Target:** Claude Code users with MCP support  
**Goal:** Real-time TER-based intervention to prevent token waste mid-session

---

## Vision

Phase 1B gives you **post-hoc analysis** and **pre-session recommendations**. Phase 2B closes the loop with **mid-session intervention**: TER monitors the active session and injects efficiency hints when it detects waste patterns forming.

**Example scenario:**
1. User starts a Claude Code session
2. Claude begins reasoning about a complex refactoring
3. TER monitors the reasoning tokens in real-time
4. After 2000 reasoning tokens with declining novelty, TER detects overthinking
5. TER injects a system prompt amendment: "You've explored thoroughly. Commit to an approach and proceed."
6. Claude pivots to implementation, saving 4000+ wasted reasoning tokens

---

## Architecture Overview

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                        Claude Code                          │
│  ┌──────────────┐         ┌─────────────┐                  │
│  │  User Prompt │────────>│   Claude    │                  │
│  └──────────────┘         │   (API)     │                  │
│                            └──────┬──────┘                  │
│                                   │                          │
│                                   │ Writes JSONL             │
│                                   v                          │
│                          ┌─────────────────┐                │
│                          │  session.jsonl  │                │
│                          └────────┬────────┘                │
│                                   │                          │
└───────────────────────────────────┼──────────────────────────┘
                                    │
                                    │ Monitors
                                    v
                        ┌───────────────────────┐
                        │   TER MCP Server      │
                        │  (localhost:3000)     │
                        ├───────────────────────┤
                        │ - JSONL watcher       │
                        │ - Rolling TER         │
                        │ - Overthinking detect │
                        │ - Drift detection     │
                        │ - Intervention logic  │
                        └───────────┬───────────┘
                                    │
                                    │ MCP Protocol
                                    │ (when intervention needed)
                                    v
                        ┌───────────────────────┐
                        │   Claude Code         │
                        │   MCP Client          │
                        ├───────────────────────┤
                        │ Injects system prompt │
                        │ amendment or tool hint│
                        └───────────────────────┘
```

### Data Flow

**Normal Operation:**
1. Claude Code writes messages to `session.jsonl` as the conversation progresses
2. TER MCP Server polls/watches the JSONL file (every 500ms)
3. Server runs `compute_rolling_ter()` on new messages
4. Server tracks metrics: TER, drift, overthinking signals
5. Server logs metrics (no intervention yet)

**Intervention Trigger:**
1. Server detects intervention condition (e.g., overthinking threshold crossed)
2. Server sends MCP message to Claude Code client
3. Claude Code injects system prompt amendment into **next** API call
4. Claude receives hint, adjusts behavior
5. TER continues monitoring, logs intervention effect

---

## MCP Server Specification

### Server Capabilities

The TER MCP server exposes these MCP tools/resources:

#### 1. **Resources** (Read-Only State)

```json
{
  "name": "ter://current-session",
  "description": "Current session TER metrics",
  "mimeType": "application/json",
  "schema": {
    "session_id": "string",
    "current_ter": "number",
    "drift": "improving|degrading|stable",
    "overthinking_detected": "boolean",
    "reasoning_efficiency": "number",
    "warning_level": "none|watch|caution|critical"
  }
}
```

#### 2. **Tools** (Callable Actions)

**`ter/analyze_session`**
- Input: `{session_path: string}`
- Output: Full TER analysis (same as `ter analyze`)
- Use case: Explicit user request for analysis

**`ter/check_intervention`**
- Input: `{session_id: string}`
- Output: `{should_intervene: boolean, intervention_type: string, message: string}`
- Use case: Claude Code polls this before each API call
- Returns intervention hints when needed

**`ter/get_budget`**
- Input: `{intent_text: string}`
- Output: Budget recommendation (same as `ter budget`)
- Use case: Pre-session budget suggestion

**`ter/record_outcome`**
- Input: `{session_id: string, actual_ter: number, intervention_applied: boolean}`
- Output: `{recorded: boolean}`
- Use case: Feedback loop for learning

#### 3. **Prompts** (Template Interventions)

**`ter/overthinking-hint`**
```
You've been reasoning for {token_count} tokens with declining novelty (current: {novelty_score}).
You've explored the problem thoroughly. Commit to an approach and proceed with implementation.
```

**`ter/reasoning-loop-hint`**
```
You appear to be restating prior reasoning. Move to action rather than continuing to explore.
```

**`ter/duplicate-tool-hint`**
```
You already ran this command at step {previous_step} with result: {result_summary}.
Consider using that result instead of re-running.
```

**`ter/context-bloat-hint`**
```
Context size has grown to {context_size} tokens. Consider summarizing or compressing the conversation history.
```

---

## Intervention Logic

### Detection Thresholds

```python
# From real_time.py and overthinking.py
INTERVENTION_THRESHOLDS = {
    # Overthinking
    "reasoning_efficiency_min": 0.70,  # Trigger if < 70% efficient
    "reasoning_tokens_min": 1500,      # Only intervene after 1500+ tokens
    "novelty_drop_consecutive": 3,     # 3 consecutive low-novelty spans
    
    # Drift
    "ter_drop_magnitude": 0.15,        # TER dropped 0.15+ over window
    "ter_drop_window": 5,              # Last 5 messages
    
    # Context bloat
    "context_size_threshold": 50000,   # 50k tokens
    "context_growth_rate": 2.0,        # Growing faster than 2x
    
    # Duplicate tools
    "tool_repeat_window": 5,           # Look back 5 tool calls
    
    # Reasoning loops
    "loop_similarity_threshold": 0.88, # Same as classifier
    "loop_min_spans": 2,               # 2 consecutive similar spans
}
```

### Intervention State Machine

```python
class InterventionState(Enum):
    MONITORING = "monitoring"           # Watching, no action
    WATCH = "watch"                     # Pattern detected, not yet actionable
    READY = "ready"                     # Ready to intervene on next opportunity
    INTERVENED = "intervened"           # Intervention sent
    COOLDOWN = "cooldown"               # Post-intervention observation period

# Transitions
MONITORING → WATCH:       threshold crossed
WATCH → READY:            pattern confirmed (not a fluke)
READY → INTERVENED:       intervention sent to Claude Code
INTERVENED → COOLDOWN:    wait N messages to see effect
COOLDOWN → MONITORING:    cooldown expired, resume normal monitoring
```

### Intervention Timing

**Critical:** Don't intervene too early or too often.

**Rules:**
1. **Minimum session length:** Only intervene after 10+ messages (avoid false positives on short sessions)
2. **Cooldown period:** After intervention, wait 5 messages before next intervention (let Claude respond)
3. **Max interventions per session:** Cap at 3 interventions (avoid being annoying)
4. **User override:** User can disable interventions via config flag

---

## Implementation Approach

### Step 1: MCP Server Core (Python)

```python
# ter_mcp_server.py
import asyncio
from pathlib import Path
from mcp.server import Server
from mcp.types import Resource, Tool, Prompt
from ter_calculator.real_time import SessionMonitor, compute_rolling_ter
from ter_calculator.overthinking import analyze_overthinking
from ter_calculator.adaptive_budget import recommend_budget

class TERMCPServer:
    def __init__(self, port=3000):
        self.server = Server("ter-efficiency-monitor")
        self.monitors = {}  # session_id -> SessionMonitor
        self.intervention_state = {}  # session_id -> InterventionState
        self.intervention_history = {}  # session_id -> list[Intervention]
        
        self._register_resources()
        self._register_tools()
        self._register_prompts()
    
    def _register_resources(self):
        @self.server.resource("ter://current-session/{session_id}")
        async def get_current_session(session_id: str):
            monitor = self.monitors.get(session_id)
            if not monitor:
                return {"error": "Session not found"}
            
            # Get latest TER signal
            state = monitor.get_state()
            return {
                "session_id": session_id,
                "current_ter": state.aggregate_ter,
                "drift": state.drift.value,
                "overthinking_detected": self._check_overthinking(session_id),
                "reasoning_efficiency": self._get_reasoning_efficiency(session_id),
                "warning_level": state.warning_level.value,
            }
    
    def _register_tools(self):
        @self.server.tool("ter/check_intervention")
        async def check_intervention(session_id: str):
            """Check if intervention is needed for this session."""
            if session_id not in self.monitors:
                # Start monitoring this session
                self._start_monitoring(session_id)
            
            should_intervene, intervention = self._evaluate_intervention(session_id)
            
            if should_intervene:
                self._record_intervention(session_id, intervention)
                return {
                    "should_intervene": True,
                    "intervention_type": intervention.type,
                    "message": intervention.message,
                    "metadata": intervention.metadata,
                }
            
            return {"should_intervene": False}
        
        @self.server.tool("ter/get_budget")
        async def get_budget(intent_text: str):
            """Get budget recommendation for a task."""
            rec = recommend_budget(intent_text)
            return {
                "complexity": rec.complexity.value,
                "model_tier": rec.model_tier.value,
                "max_thinking_tokens": rec.max_thinking_tokens,
                "estimated_cost_usd": rec.estimated_cost_usd,
                "reasoning": rec.reasoning,
            }
        
        @self.server.tool("ter/record_outcome")
        async def record_outcome(
            session_id: str,
            actual_ter: float,
            intervention_applied: bool
        ):
            """Record session outcome for learning."""
            # Update historical budget analyzer
            self._update_history(session_id, actual_ter, intervention_applied)
            return {"recorded": True}
    
    def _register_prompts(self):
        @self.server.prompt("ter/overthinking-hint")
        async def overthinking_hint(session_id: str):
            """Generate overthinking intervention message."""
            monitor = self.monitors[session_id]
            state = monitor.get_state()
            
            return f"""You've been reasoning for {state.reasoning_tokens} tokens with declining novelty.
You've explored the problem thoroughly. Commit to an approach and proceed with implementation."""
        
        # Similar for other intervention types...
    
    def _evaluate_intervention(self, session_id: str) -> tuple[bool, Intervention]:
        """Core intervention logic."""
        monitor = self.monitors[session_id]
        state = monitor.get_state()
        current_state = self.intervention_state.get(session_id, InterventionState.MONITORING)
        
        # Don't intervene during cooldown
        if current_state == InterventionState.COOLDOWN:
            if self._cooldown_expired(session_id):
                self.intervention_state[session_id] = InterventionState.MONITORING
            return False, None
        
        # Check max interventions
        if len(self.intervention_history.get(session_id, [])) >= 3:
            return False, None
        
        # Check overthinking
        if self._check_overthinking(session_id):
            intervention = Intervention(
                type="overthinking",
                message=self._get_overthinking_message(session_id),
                metadata={"reasoning_tokens": state.reasoning_tokens}
            )
            self.intervention_state[session_id] = InterventionState.INTERVENED
            return True, intervention
        
        # Check TER drift
        if state.drift == DriftDirection.DEGRADING and state.drift_magnitude > 0.15:
            intervention = Intervention(
                type="ter_drift",
                message=f"Session efficiency has dropped {state.drift_magnitude:.1%}. Consider simplifying your approach.",
                metadata={"drift_magnitude": state.drift_magnitude}
            )
            self.intervention_state[session_id] = InterventionState.INTERVENED
            return True, intervention
        
        # Check other patterns...
        
        return False, None
    
    async def start(self):
        """Start the MCP server."""
        await self.server.run(host="localhost", port=3000)
```

### Step 2: Claude Code Integration

Claude Code would need to:

1. **Detect TER MCP Server** (via MCP discovery)
2. **Register connection** to `localhost:3000`
3. **Poll before each API call:**
   ```typescript
   // Before sending message to Claude API
   const intervention = await mcp.call("ter/check_intervention", {
     session_id: currentSessionId
   });
   
   if (intervention.should_intervene) {
     // Inject system prompt amendment
     systemPrompt += `\n\n${intervention.message}`;
     
     // Log intervention
     console.log(`[TER] Intervening: ${intervention.intervention_type}`);
   }
   ```
4. **Record outcome** after session:
   ```typescript
   // After session completes
   await mcp.call("ter/record_outcome", {
     session_id: sessionId,
     actual_ter: finalTER,
     intervention_applied: interventionsApplied.length > 0
   });
   ```

### Step 3: Configuration

User config in `~/.claude/ter-mcp.json`:

```json
{
  "enabled": true,
  "intervention_mode": "auto",  // "auto" | "manual" | "disabled"
  "thresholds": {
    "overthinking_efficiency_min": 0.70,
    "ter_drift_threshold": 0.15,
    "context_bloat_threshold": 50000
  },
  "intervention_limits": {
    "max_per_session": 3,
    "cooldown_messages": 5,
    "min_session_length": 10
  },
  "notification_style": "inline",  // "inline" | "system" | "none"
  "log_level": "info"
}
```

---

## Intervention Types & Messages

### 1. Overthinking
**Trigger:** Reasoning efficiency < 70%, 1500+ tokens, declining novelty  
**Message:**
```
[TER Efficiency Hint] You've been reasoning for 2,100 tokens with declining novelty (48% efficiency).
You've explored thoroughly. Commit to an approach and proceed with implementation.
```

### 2. TER Drift (Degrading)
**Trigger:** TER dropped 15%+ over last 5 messages  
**Message:**
```
[TER Efficiency Hint] Session efficiency has dropped 18% over recent messages.
Consider simplifying your approach or breaking the task into smaller steps.
```

### 3. Reasoning Loop
**Trigger:** 2+ consecutive similar reasoning spans (similarity > 0.88)  
**Message:**
```
[TER Efficiency Hint] You appear to be restating prior reasoning.
Move to action rather than continuing to explore.
```

### 4. Duplicate Tool Call
**Trigger:** Same tool call within 5-step window  
**Message:**
```
[TER Efficiency Hint] You already ran this command at message #12.
Result: <summary>. Consider using that result instead of re-running.
```

### 5. Context Bloat
**Trigger:** Context > 50k tokens, super-linear growth  
**Message:**
```
[TER Efficiency Hint] Context has grown to 52,000 tokens (2.3x growth rate).
Consider summarizing earlier parts of the conversation to maintain efficiency.
```

---

## Metrics & Observability

### What to Track

**Per-Session Metrics:**
- Intervention count
- Intervention types
- TER before/after each intervention
- User acceptance rate (did user follow hint?)
- Session outcome (final TER)

**Aggregate Metrics:**
- Average TER improvement from interventions
- Intervention precision (true positives / total interventions)
- False positive rate (interventions that hurt TER)
- Token savings (estimated)
- Cost savings (estimated)

**Dashboard:**
```
TER MCP Server - Live Metrics
═══════════════════════════════════════

Active Sessions: 3
Total Interventions Today: 12
  - Overthinking: 7
  - TER Drift: 3
  - Duplicate Tool: 2

Impact:
  Avg TER Improvement: +0.08
  Estimated Token Savings: 15,420
  Estimated Cost Savings: $0.42

Intervention Precision: 91.7% (11/12 helpful)
```

### Logging

```json
{
  "timestamp": "2026-05-05T14:23:45Z",
  "session_id": "abc123",
  "event": "intervention",
  "type": "overthinking",
  "trigger": {
    "reasoning_tokens": 2100,
    "efficiency": 0.48,
    "novelty_score": 0.12
  },
  "action": {
    "message": "You've been reasoning for 2,100 tokens...",
    "injected": true
  },
  "context": {
    "message_index": 15,
    "session_ter_before": 0.72,
    "previous_interventions": 1
  }
}
```

---

## Safety & User Experience

### Safety Mechanisms

1. **Rate Limiting**
   - Max 3 interventions per session
   - 5-message cooldown between interventions
   - No interventions in first 10 messages

2. **Confidence Thresholds**
   - Only intervene on high-confidence signals
   - Require pattern confirmation (not single-message anomaly)
   - Combine multiple signals (e.g., overthinking + drift)

3. **User Control**
   - `intervention_mode: "manual"` - show suggestion, user approves
   - `intervention_mode: "disabled"` - monitoring only, no actions
   - Per-intervention-type toggles

4. **Graceful Degradation**
   - If MCP connection fails, fall back to monitoring only
   - Never block Claude Code operation
   - Log errors, don't crash

### User Experience

**Good UX:**
```
[Claude Code UI]
┌─────────────────────────────────────────────┐
│ 💡 TER Efficiency Suggestion                │
│                                             │
│ You've been reasoning for 2,100 tokens     │
│ with declining novelty. Consider moving    │
│ to implementation.                          │
│                                             │
│ [Apply Hint] [Dismiss] [Disable for this  │
│              session]                       │
└─────────────────────────────────────────────┘
```

**Bad UX:**
```
ERROR: TER INTERVENTION FAILED
System prompt injection blocked
Session terminated
```

### Notification Styles

**1. Inline (Recommended)**
- Appears as a system message in the conversation
- Looks like Claude's own reflection
- Non-intrusive

**2. System Notification**
- OS-level notification
- For critical issues only (e.g., runaway session)

**3. Status Bar**
- Shows TER metric in Claude Code status bar
- Subtle, always visible
- Color-coded: green (>0.7), yellow (0.4-0.7), red (<0.4)

---

## Deployment Model

### Local-First (Phase 1)

```
User Machine:
  - Claude Code (MCP client)
  - TER MCP Server (Python, localhost:3000)
  - Shared JSONL files
```

**Pros:**
- No network dependency
- Fast (local I/O)
- Private (data never leaves machine)

**Cons:**
- User must install/run TER server
- No cross-device state

### Cloud-Assisted (Phase 2)

```
User Machine:
  - Claude Code (MCP client)
  
Cloud:
  - TER MCP Server (managed instance)
  - Historical data storage
  - Cross-session learning
```

**Pros:**
- Zero-config for users
- Cross-device learning
- Better recommendations (more data)

**Cons:**
- Latency (network roundtrip)
- Privacy concerns (upload session data)
- Cost (server hosting)

**Hybrid (Best of Both):**
- Local server for real-time intervention
- Cloud sync for historical learning
- User controls what data syncs

---

## Development Roadmap

### Milestone 1: Monitoring Only (1 week)
- [ ] Build TER MCP server skeleton
- [ ] Implement JSONL watching
- [ ] Expose `ter://current-session` resource
- [ ] Test with Claude Code MCP client
- [ ] No interventions yet, just monitoring

**Success:** Claude Code can read current TER from MCP server

### Milestone 2: Manual Interventions (1 week)
- [ ] Implement intervention detection logic
- [ ] Add `ter/check_intervention` tool
- [ ] Build intervention prompt templates
- [ ] Add `intervention_mode: "manual"` support
- [ ] User approval UI in Claude Code

**Success:** User can manually approve TER suggestions

### Milestone 3: Auto Interventions (1 week)
- [ ] Enable `intervention_mode: "auto"`
- [ ] Implement state machine (monitoring → intervention → cooldown)
- [ ] Add rate limiting & safety checks
- [ ] Build intervention logging
- [ ] Add user config file support

**Success:** Fully automated intervention with safety rails

### Milestone 4: Learning & Optimization (2 weeks)
- [ ] Implement `ter/record_outcome` tool
- [ ] Track intervention effectiveness
- [ ] Adjust thresholds based on outcomes
- [ ] Build metrics dashboard
- [ ] A/B test intervention messages

**Success:** System learns which interventions work best

---

## Open Questions

### Technical

1. **MCP Protocol Maturity:** Is MCP stable enough for production use? (As of 2026-05, yes - widely adopted)

2. **JSONL File Watching:** Poll vs inotify? (Recommend: inotify on Linux/Mac, poll on Windows with 500ms interval)

3. **Concurrent Sessions:** How to handle multiple active sessions? (Use `session_id` as key, one monitor per session)

4. **Session Discovery:** How does server know when a new session starts? (Watch `.claude/projects/*/sessions/*.jsonl` pattern, detect new files)

### UX

1. **Intervention Tone:** Directive vs suggestive? ("Commit to an approach" vs "Consider committing to an approach")
   - **Recommendation:** Suggestive for first intervention, more directive if ignored

2. **Visibility:** Should user always see interventions or only in verbose mode?
   - **Recommendation:** Always visible but subtle (inline, not modal)

3. **Override Mechanism:** How easy should it be to disable?
   - **Recommendation:** Session-level toggle in UI, global toggle in config

### Research

1. **Intervention Timing:** When is the optimal moment to intervene? (Need to collect data)

2. **Message Effectiveness:** Which intervention messages actually change behavior? (A/B test)

3. **False Positive Rate:** How often do we intervene when we shouldn't? (Target: <10%)

---

## Success Criteria

### Phase 2B is successful if:

1. **Effectiveness**
   - Average thinking token reduction: 30%+ on simple tasks
   - No quality degradation (TER stays above baseline)
   - Intervention precision: >90%

2. **Adoption**
   - Users keep interventions enabled (not disabled)
   - Positive user feedback
   - Measurable cost savings reported

3. **Reliability**
   - MCP server uptime: >99%
   - Intervention latency: <100ms
   - Zero false positives causing harm

4. **Learning**
   - System improves over time (fewer bad interventions)
   - Historical data informs better thresholds
   - Personalization per user/project

---

## Next Steps (After Reading This Doc)

1. **Validate Assumptions**
   - Use `ter watch` for a week on your own sessions
   - Manually note where you'd want intervention
   - Check if thresholds make sense

2. **Prototype MCP Server**
   - Build basic server with monitoring only
   - Test Claude Code MCP client integration
   - Verify JSONL watching works reliably

3. **Design User UI**
   - Mock up intervention notification in Claude Code
   - Test with users (show/hide vs always-on)
   - Refine messaging tone

4. **Build Milestone 1**
   - Implement monitoring-only version
   - Deploy locally, dogfood for a week
   - Iterate based on experience

---

## References

- [MCP Protocol Spec](https://spec.modelcontextprotocol.io/)
- [Claude Code MCP Integration Guide](https://github.com/anthropics/claude-code/docs/mcp.md)
- TER Phase 1B modules: `real_time.py`, `overthinking.py`, `adaptive_budget.py`
- Research: IARS, SelfBudgeter, TALE (see `plan.md`)

---

**This architecture is ready to implement when you are.** Start with Milestone 1 (monitoring only), validate the approach, then progressively add intervention logic. The MCP layer keeps it clean, testable, and decoupled from Claude Code internals.
