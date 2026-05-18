# TER Calculator -- Architecture

**Date**: 2026-05-15 | **Branch**: `feature/context-orchestrator`

---

## 1. System Overview

TER (Token Efficiency Ratio) is a Python 3.11+ CLI tool that measures how
efficiently Claude Code sessions use output tokens. It ingests JSONL session
transcripts, classifies every output token span as **aligned** (contributing to
the user's intent) or **waste** (redundant reasoning, duplicate tool calls,
over-explanation), computes a weighted efficiency score, and surfaces session
economics.

The system now includes a **Context Orchestrator** subsystem that manages
cross-session context fragments. The orchestrator decomposes session context into
content-addressable fragments, organises them in a directed acyclic graph,
applies knapsack-based budget optimisation, composes reference-based delta
prompts, and coordinates version consistency across concurrent sessions.

### External Dependencies

| Dependency | Role |
|---|---|
| `sentence-transformers` | Embedding model (`all-MiniLM-L6-v2`, 384-dim, ~22 MB) |
| `numpy` | Vectorised cosine similarity, knapsack DP table |
| `rich` | Terminal formatting (tables, colour-coded scores) |
| `sqlite3` (stdlib) | Fragment store and edge persistence |

---

## 2. Module Dependency Graph

All source lives under `src/ter_calculator/`. Every module imports `models.py`.
Only `cli.py` fans out to the full set.

```mermaid
graph BT
    models["models.py<br/>(dataclasses, enums)"]

    subgraph core["Core Pipeline"]
        loader["loader.py"]
        intent["intent.py"]
        classifier["classifier.py"]
        waste["waste.py"]
        compute["compute.py"]
        formatter["formatter.py"]
        compare["compare.py"]
        analyze["analyze_pipeline.py"]
        cfgparse["config_parse.py"]
        sessrpt["session_report.py"]
    end

    subgraph bridge["Bridge Modules"]
        rt["real_time.py"]
        ab["adaptive_budget.py"]
        cm["cost_model.py"]
        ot["overthinking.py"]
    end

    subgraph improvement["Improvement Modules"]
        emb["embedding_cache.py"]
        tok["token_counting.py"]
        iex["intent_extraction.py"]
        wd["waste_detectors.py"]
        fb["feedback.py"]
        pl["plugins.py"]
        val["validation.py"]
        acc["acceleration.py"]
    end

    subgraph inputa["Input Analysis"]
        ia["input_analysis.py"]
        ec["economics.py"]
    end

    subgraph orchestrator["Context Orchestrator"]
        fs["fragment_store.py"]
        cg["context_graph.py"]
        bo["budget_optimizer.py"]
        dc["delta_composer.py"]
        con["consistency.py"]
    end

    cli["cli.py<br/>(entry point)"]

    %% Everything depends on models
    loader --> models
    intent --> models
    classifier --> models
    waste --> models
    compute --> models
    formatter --> models
    compare --> models
    analyze --> models
    cfgparse --> models
    sessrpt --> models
    rt --> models
    ab --> models
    cm --> models
    ot --> models
    emb --> models
    tok --> models
    iex --> models
    wd --> models
    fb --> models
    pl --> models
    val --> models
    acc --> models
    ia --> models
    ec --> models
    fs --> models
    cg --> models
    bo --> models
    dc --> models
    con --> models

    %% Core inter-module edges
    intent --> emb
    intent --> iex
    classifier --> emb
    classifier --> tok
    waste --> wd
    waste --> ot
    loader --> val
    compute --> fb
    compute --> cm
    rt --> acc

    %% Context Orchestrator internal chain
    cg --> fs
    bo --> fs
    bo --> cg
    bo --> classifier
    dc --> fs
    con --> fs
    con --> cg

    %% CLI fans out
    cli --> loader
    cli --> intent
    cli --> classifier
    cli --> waste
    cli --> compute
    cli --> compare
    cli --> formatter
    cli --> pl
    cli --> val
    cli --> acc
    cli --> fb
    cli --> rt
    cli --> ab
    cli --> cm

    style models fill:#e1f5fe
    style cli fill:#fff3e0
    style fs fill:#f3e5f5
    style cg fill:#f3e5f5
    style bo fill:#f3e5f5
    style dc fill:#f3e5f5
    style con fill:#f3e5f5
    style emb fill:#e8f5e9
    style tok fill:#e8f5e9
    style iex fill:#e8f5e9
    style wd fill:#e8f5e9
    style fb fill:#e8f5e9
    style pl fill:#e8f5e9
    style val fill:#e8f5e9
    style acc fill:#e8f5e9
    style rt fill:#fce4ec
    style ot fill:#fce4ec
    style ab fill:#fce4ec
    style cm fill:#fce4ec
```

### Context Orchestrator Internal Dependencies

The five orchestrator modules form their own dependency sub-graph:

```
fragment_store.py  (no orchestrator deps -- foundation)
    |
    +---> context_graph.py       (imports fragment_store for DB path sharing)
    |         |
    |         +---> budget_optimizer.py  (imports fragment_store + context_graph + classifier)
    |
    +---> delta_composer.py      (imports fragment_store)
    |
    +---> consistency.py         (imports fragment_store + context_graph)
```

No circular dependencies exist. `fragment_store` is the leaf; all other
orchestrator modules depend on it.

---

## 3. Data Flow Diagrams

### 3a. Analysis Pipeline

The primary pipeline transforms a JSONL session file into a TER report.

```mermaid
flowchart LR
    subgraph S1["1. LOAD"]
        A1["Read JSONL"] --> A2["Dedup by requestId"]
        A2 --> A3["Build Messages"]
        A3 --> A4["Segment Spans<br/>thinking = reasoning<br/>tool_use/result = tool_use<br/>text = generation"]
    end

    subgraph S2["2. INTENT"]
        B1["Combine user prompts"] --> B2["Generate 384-dim<br/>embedding"]
        B2 --> B3["Score confidence"]
    end

    subgraph S3["3. CLASSIFY"]
        C1["Embed each span"] --> C2["Cosine similarity<br/>vs intent vector"]
        C2 --> C3["Apply thresholds<br/>sim=0.40 conf=0.75"]
        C3 --> C4["Label:<br/>aligned / waste<br/>per phase"]
    end

    subgraph S4["4. DETECT"]
        D1["Reasoning loops"]
        D2["Duplicate tool calls"]
        D3["Context restatement"]
    end

    subgraph S5["5. COMPUTE"]
        E1["Per-phase TER<br/>aligned / total"] --> E2["Weighted aggregate<br/>R:0.3  T:0.4  G:0.3"]
        E2 --> E3["Raw ratio +<br/>token counts"]
    end

    subgraph S6["6. OUTPUT"]
        F1["Text: Rich tables"]
        F2["JSON: structured"]
    end

    S1 --> S2 --> S3 --> S4 --> S5 --> S6
```

### 3b. Context Orchestrator Pipeline

The orchestrator pipeline manages cross-session context reuse.

```mermaid
flowchart LR
    subgraph SHARD["1. SHARD"]
        X1["TokenSpans from<br/>analysis pipeline"] --> X2["Normalise + SHA-256<br/>content hash"]
        X2 --> X3["Deduplicate against<br/>existing store"]
        X3 --> X4["Embed new fragments<br/>(all-MiniLM-L6-v2)"]
    end

    subgraph STORE["2. STORE"]
        Y1["FragmentStore<br/>(SQLite)"] --> Y2["Content-addressable<br/>put / get / gc"]
    end

    subgraph GRAPH["3. GRAPH"]
        Z1["Build DAG nodes<br/>from fragments"] --> Z2["Infer edges:<br/>dependency<br/>derivation<br/>co-occurrence"]
        Z2 --> Z3["Persist edges<br/>to SQLite"]
    end

    subgraph BUDGET["4. BUDGET"]
        W1["Score fragments<br/>vs intent vector"] --> W2["Prune redundant<br/>(sim > 0.85)"]
        W2 --> W3["Knapsack optimisation<br/>DP or greedy"]
        W3 --> W4["OptimizationResult:<br/>selected IDs +<br/>token budget used"]
    end

    subgraph DELTA["5. DELTA"]
        V1["Build PromptTemplate<br/>with fragment placeholders"] --> V2["Diff against<br/>local LRU cache"]
        V2 --> V3["Transmit only<br/>uncached fragments"]
        V3 --> V4["DeltaPrompt:<br/>compression ratio +<br/>tokens saved"]
    end

    subgraph CONSIST["6. CONSISTENCY"]
        U1["Register session<br/>fragment versions"] --> U2["Detect version skew<br/>across sessions"]
        U2 --> U3["Resolve: strict block<br/>or relaxed warning"]
    end

    SHARD --> STORE --> GRAPH --> BUDGET --> DELTA --> CONSIST
```

---

## 4. Entity Relationship Diagram

### 4a. Core Analysis Models

```mermaid
erDiagram
    Session ||--o{ Message : contains
    Message ||--o{ ContentBlock : contains
    Message ||--o| TokenUsage : has
    Session ||--|| IntentVector : "extract intent"
    Session ||--o{ TokenSpan : "segment into"
    TokenSpan ||--|| ClassifiedSpan : "classify into"
    ClassifiedSpan }o--o{ WastePattern : "detected from"
    ClassifiedSpan }o--|| TERResult : "aggregated into"
    WastePattern }o--|| TERResult : "included in"
    IntentVector ||--|| TERResult : "referenced by"
    TERResult ||--o| SessionEconomics : "includes"
    TERResult ||--o| InputAnalysis : "includes"
    SessionEconomics ||--|| CostModel : "priced by"

    Session {
        string session_id PK
        string file_path
        datetime timestamp
        int total_tokens
    }

    Message {
        string uuid PK
        string role
        string parent_uuid FK
        string request_id
        string stop_reason
    }

    ContentBlock {
        string block_type
        string text
        string tool_name
        dict tool_input
    }

    TokenUsage {
        int input_tokens
        int output_tokens
        int cache_creation_input_tokens
        int cache_read_input_tokens
    }

    TokenSpan {
        string text
        enum phase
        int position
        int token_count
        string source_message_uuid FK
        vector embedding
    }

    IntentVector {
        string text
        vector embedding
        float confidence
    }

    ClassifiedSpan {
        enum label
        float confidence
        float cosine_similarity
    }

    WastePattern {
        string pattern_type
        string description
        int start_position
        int end_position
        int tokens_wasted
    }

    TERResult {
        string session_id FK
        float aggregate_ter
        float raw_ratio
        int total_tokens
        int aligned_tokens
        int waste_tokens
    }

    CostModel {
        float input_rate
        float output_rate
        float cache_read_rate
        float cache_write_rate
    }

    SessionEconomics {
        int total_input_tokens
        int total_output_tokens
        float cache_hit_rate
        float estimated_cost_usd
        float estimated_waste_cost_usd
    }

    InputAnalysis {
        object token_breakdown
        object prompt_similarity
        object intent_drift
        object prompt_response_alignment
    }
```

### 4b. Context Orchestrator Models

```mermaid
erDiagram
    Fragment ||--o{ FragmentEdge : "source or target"
    Fragment ||--o{ FragmentNode : "represented by"
    Fragment ||--o{ ScoredFragment : "scored as"
    ScoredFragment }o--|| OptimizationResult : "selected into"
    Fragment }o--o| PromptTemplate : "referenced by"
    PromptTemplate ||--|| DeltaPrompt : "composed into"
    DeltaPrompt ||--|| FragmentManifest : "accompanied by"
    Fragment ||--o{ FragmentVersion : "versioned as"
    FragmentVersion }o--o{ VersionSkew : "detected in"
    VersionSkew ||--|| ConsistencyAction : "resolved by"
    Fragment ||--o{ InvalidationEvent : "invalidated by"

    Fragment {
        string id PK
        string text
        int token_count
        enum phase
        string origin_session
        float created_at
        int ttl_seconds
        vector embedding
    }

    FragmentNode {
        string fragment_id FK
        float creation_timestamp
        string origin_session
        int token_weight
        int staleness_ttl
    }

    FragmentEdge {
        string source_id FK
        string target_id FK
        enum edge_type
        float weight
    }

    ScoredFragment {
        string fragment_id FK
        float relevance_score
        int token_cost
        enum phase
        bool is_cached
    }

    OptimizationResult {
        list selected_fragment_ids
        int total_tokens
        float total_relevance
        int budget_used
        int budget_ceiling
        int pruned_count
        string reasoning
    }

    PromptTemplate {
        string template_text
        list required_fragment_ids
    }

    FragmentManifest {
        list fragment_ids
        int total_tokens
        int cache_hits
        int cache_misses
    }

    DeltaPrompt {
        object template
        object manifest
        list delta_fragments
        int total_tokens_saved
        float compression_ratio
    }

    InvalidationEvent {
        string fragment_id FK
        float timestamp
        string reason
    }

    FragmentVersion {
        string fragment_id FK
        int version
        string content_hash
        float timestamp
    }

    VersionSkew {
        string fragment_id FK
        list sessions_involved
        dict versions_seen
        string severity
    }

    ConsistencyAction {
        bool block
        string message
        list refresh_fragment_ids
    }
```

---

## 5. Storage Architecture

TER uses three storage layers with distinct lifecycles.

```
Storage Layer          Format       Location                             Lifecycle
---------------------  -----------  -----------------------------------  --------------------
Session transcripts    JSONL        ~/.claude/projects/<hash>/<uuid>.jsonl  Read-only input
Fragment store + edges SQLite (WAL) ~/.cache/ter/fragments.db            Append + GC (TTL)
Analysis history       JSON          ~/.cache/ter/history.json            Append
Budget history         JSON          ~/.cache/ter/budget_history.json     Append
Embedding cache        Binary        ~/.cache/ter/embeddings/             Content-addressed
```

### SQLite Schema (`fragments.db`)

The fragment store and context graph share a single SQLite database with WAL
journaling for concurrent read access.

**Table: `fragments`**

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PK | SHA-256 of normalised text |
| `text` | TEXT | NFC-normalised, whitespace-collapsed |
| `embedding` | BLOB | 384 x float32 = 1,536 bytes |
| `token_count` | INTEGER | Heuristic or calibrated count |
| `phase` | TEXT | reasoning / tool_use / generation |
| `origin_session` | TEXT | Session ID that created this fragment |
| `created_at` | REAL | Unix epoch timestamp |
| `ttl_seconds` | INTEGER | Default 3600 (1 hour) |

**Table: `edges`**

| Column | Type | Notes |
|---|---|---|
| `source_id` | TEXT | FK to fragments.id |
| `target_id` | TEXT | FK to fragments.id |
| `edge_type` | TEXT | dependency / derivation / co_occurrence |
| `weight` | REAL | Default 1.0 |
| PK | | (source_id, target_id, edge_type) |

**Table: `schema_version`**

Single-row table tracking migration version (currently version 1).

### Garbage Collection

- `FragmentStore.gc(max_age_hours)` deletes rows where
  `created_at + ttl_seconds < cutoff`.
- `ContextGraph.prune_stale(max_age_hours)` removes orphaned nodes and their
  edges from both the in-memory adjacency list and SQLite.

---

## 6. Embedding and Similarity

### Model

- **Model**: `all-MiniLM-L6-v2` via `sentence-transformers`
- **Dimensionality**: 384
- **Download size**: ~22 MB (cached locally after first run)
- **Inference**: CPU by default; GPU auto-detected when available

### Similarity Function

Cosine similarity between two vectors **a** and **b**:

```
cos(a, b) = dot(a, b) / (norm(a) * norm(b))
```

Implemented in `classifier.py` using NumPy vectorised operations.

### Content-Addressable Hashing

Fragment identity is derived from a SHA-256 hash of the text after
normalisation:

1. Apply Unicode NFC normalisation.
2. Collapse all whitespace runs to single spaces.
3. Compute `hashlib.sha256(normalized.encode("utf-8")).hexdigest()`.

Two fragments with identical normalised text always produce the same ID,
enabling cross-session deduplication without comparing embeddings.

### Thresholds

| Parameter | Default | Used In |
|---|---|---|
| Alignment similarity | 0.40 | `classifier.py` -- span is aligned if cosine sim >= threshold |
| Alignment confidence | 0.75 | `classifier.py` -- minimum classifier confidence |
| Redundancy pruning | 0.85 | `budget_optimizer.py` -- prune near-duplicate fragments |
| Relevance floor | 0.10 | `budget_optimizer.py` -- discard low-relevance fragments before knapsack |

### Budget Optimisation Strategy

The `budget_optimizer.py` module uses a two-tier approach:

1. **Dynamic programming knapsack** when fragment count is at most 500 (exact
   solution, quantised to 100-token granularity).
2. **Greedy fractional knapsack** when fragment count exceeds 500 (approximation,
   ranked by relevance-per-token).

Phase weights (reasoning 0.3, tool_use 0.4, generation 0.3) are applied as
multipliers on the cosine similarity score before optimisation.
