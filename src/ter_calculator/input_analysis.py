"""Input analysis: user-centric vs model-centric token breakdown and prompt similarity."""

from __future__ import annotations

from .classifier import cosine_similarity
from .intent import embed_texts
from .models import (
    InputAnalysis,
    IntentDrift,
    IntentDriftStep,
    PromptPair,
    PromptResponseAlignment,
    PromptResponsePair,
    PromptSimilarityResult,
    Session,
    TokenBreakdown,
)


def analyze_input(
    session: Session,
    similarity_threshold: float = 0.75,
    alignment_threshold: float = 0.3,
) -> InputAnalysis:
    """Full input analysis: token breakdown, prompt similarity, drift, and alignment."""
    breakdown = compute_token_breakdown(session)
    similarity = compute_prompt_similarity(
        session.user_prompts, similarity_threshold=similarity_threshold
    )
    drift = compute_intent_drift(session.user_prompts)
    alignment = compute_prompt_response_alignment(
        session, alignment_threshold=alignment_threshold
    )
    return InputAnalysis(
        token_breakdown=breakdown,
        prompt_similarity=similarity,
        intent_drift=drift,
        prompt_response_alignment=alignment,
    )


def compute_token_breakdown(session: Session) -> TokenBreakdown:
    """Categorize all tokens by origin (user vs model).

    User-centric tokens:
      - user message text blocks
      - tool_result blocks (content returned to the model on behalf of the user)

    Model-centric tokens:
      - thinking blocks → reasoning
      - tool_use blocks → tool invocations
      - text blocks in assistant messages → generation
    """
    user_input = 0
    user_result = 0
    model_reasoning = 0
    model_tool = 0
    model_generation = 0

    for msg in session.messages:
        for block in msg.content_blocks:
            text = block.text or ""
            token_count = max(1, len(text) // 4) if text else 0

            if msg.role == "user":
                if block.block_type == "tool_result":
                    user_result += token_count
                else:
                    user_input += token_count
            elif msg.role == "assistant":
                if block.block_type == "thinking":
                    model_reasoning += token_count
                elif block.block_type in ("tool_use", "tool_result"):
                    model_tool += token_count
                elif block.block_type == "text":
                    model_generation += token_count

    total_user = user_input + user_result
    total_model = model_reasoning + model_tool + model_generation
    total = total_user + total_model

    return TokenBreakdown(
        user_input_tokens=user_input,
        user_result_tokens=user_result,
        model_reasoning_tokens=model_reasoning,
        model_tool_tokens=model_tool,
        model_generation_tokens=model_generation,
        total_user_tokens=total_user,
        total_model_tokens=total_model,
        user_ratio=round(total_user / total, 4) if total > 0 else 0.0,
    )


def compute_prompt_similarity(
    prompts: list[str],
    similarity_threshold: float = 0.75,
) -> PromptSimilarityResult:
    """Compute pairwise similarity between user prompts.

    Flags prompt pairs above the similarity threshold as semantically
    similar (potential redundant asks). Returns a redundancy score
    representing the fraction of prompts involved in at least one
    near-duplicate pair.
    """
    if len(prompts) <= 1:
        return PromptSimilarityResult(
            similarity_matrix=[[1.0]] if prompts else [],
            similar_pairs=[],
            prompt_redundancy_score=0.0,
            prompt_count=len(prompts),
        )

    embeddings = embed_texts(prompts)

    n = len(prompts)
    matrix: list[list[float]] = []
    similar_pairs: list[PromptPair] = []
    redundant_indices: set[int] = set()

    for i in range(n):
        row: list[float] = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                sim = cosine_similarity(embeddings[i], embeddings[j])
                row.append(round(sim, 4))

                if j > i and sim >= similarity_threshold:
                    similar_pairs.append(PromptPair(
                        prompt_a_index=i,
                        prompt_b_index=j,
                        similarity=round(sim, 4),
                        prompt_a_text=prompts[i],
                        prompt_b_text=prompts[j],
                    ))
                    redundant_indices.add(i)
                    redundant_indices.add(j)
        matrix.append(row)

    similar_pairs.sort(key=lambda p: p.similarity, reverse=True)
    redundancy_score = len(redundant_indices) / n if n > 0 else 0.0

    return PromptSimilarityResult(
        similarity_matrix=matrix,
        similar_pairs=similar_pairs,
        prompt_redundancy_score=round(redundancy_score, 4),
        prompt_count=n,
    )


def compute_intent_drift(prompts: list[str]) -> IntentDrift:
    """Track how user intent evolves between consecutive prompts.

    Each consecutive pair is classified:
      - convergent (≥0.6): user is refining/repeating the same ask
      - divergent (≤0.4): user moved to a new topic
      - evolving (between): gradual shift

    Overall trajectory summarises the session pattern.
    """
    if len(prompts) <= 1:
        return IntentDrift(steps=[], overall_trajectory="stable", average_drift=0.0)

    embeddings = embed_texts(prompts)

    steps: list[IntentDriftStep] = []
    for i in range(len(prompts) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        sim = round(sim, 4)

        if sim >= 0.6:
            drift_type = "convergent"
        elif sim <= 0.4:
            drift_type = "divergent"
        else:
            drift_type = "evolving"

        steps.append(IntentDriftStep(
            from_index=i,
            to_index=i + 1,
            similarity=sim,
            drift_type=drift_type,
        ))

    avg = sum(s.similarity for s in steps) / len(steps)
    trajectory = _classify_trajectory(steps)

    return IntentDrift(
        steps=steps,
        overall_trajectory=trajectory,
        average_drift=round(avg, 4),
    )


def _classify_trajectory(steps: list[IntentDriftStep]) -> str:
    """Classify overall session trajectory from individual drift steps."""
    if not steps:
        return "stable"

    counts = {"convergent": 0, "divergent": 0, "evolving": 0}
    for s in steps:
        counts[s.drift_type] += 1

    n = len(steps)
    if counts["evolving"] == n:
        return "stable"
    if counts["convergent"] > n / 2:
        return "convergent"
    if counts["divergent"] > n / 2:
        return "divergent"
    return "mixed"


def compute_prompt_response_alignment(
    session: Session,
    alignment_threshold: float = 0.3,
) -> PromptResponseAlignment:
    """Measure how well each model response aligns with the user's prompt.

    Walks messages in order, pairs each real user text prompt with the
    text generation blocks from the following assistant message(s).
    Tool-result user messages are skipped (they're intermediate, not asks).
    """
    pairs = _extract_prompt_response_pairs(session)

    if not pairs:
        return PromptResponseAlignment(
            pairs=[], average_alignment=0.0, low_alignment_count=0
        )

    # Batch-embed all prompts and responses together for efficiency.
    all_texts = [p[1] for p in pairs] + [p[2] for p in pairs]
    embeddings = embed_texts(all_texts)

    n = len(pairs)
    result_pairs: list[PromptResponsePair] = []
    low_count = 0

    for i, (idx, prompt_text, response_text) in enumerate(pairs):
        prompt_emb = embeddings[i]
        response_emb = embeddings[n + i]
        alignment = round(cosine_similarity(prompt_emb, response_emb), 4)

        result_pairs.append(PromptResponsePair(
            prompt_index=idx,
            prompt_text=prompt_text,
            response_text=response_text,
            alignment=alignment,
        ))
        if alignment < alignment_threshold:
            low_count += 1

    avg = sum(p.alignment for p in result_pairs) / len(result_pairs)

    return PromptResponseAlignment(
        pairs=result_pairs,
        average_alignment=round(avg, 4),
        low_alignment_count=low_count,
    )


def _extract_prompt_response_pairs(
    session: Session,
) -> list[tuple[int, str, str]]:
    """Extract (prompt_index, prompt_text, response_text) tuples.

    A real user prompt is a user message containing a text block
    (not just a tool_result). The response is the concatenated text
    generation blocks from subsequent assistant messages until the
    next real user prompt.
    """
    pairs: list[tuple[int, str, str]] = []
    prompt_index = 0
    current_prompt: str | None = None
    response_parts: list[str] = []

    for msg in session.messages:
        if msg.role == "user":
            # Check if this is a real user prompt (has text blocks, not just tool_result).
            user_texts = [
                b.text for b in msg.content_blocks
                if b.block_type == "text" and b.text
            ]
            if user_texts:
                # Flush previous pair if we have one.
                if current_prompt is not None and response_parts:
                    pairs.append((
                        prompt_index,
                        current_prompt,
                        " ".join(response_parts),
                    ))
                    prompt_index += 1
                    response_parts = []

                current_prompt = " ".join(user_texts)
            # tool_result-only messages: skip, don't reset current_prompt

        elif msg.role == "assistant" and current_prompt is not None:
            # Collect text generation blocks as the response.
            for block in msg.content_blocks:
                if block.block_type == "text" and block.text:
                    response_parts.append(block.text)

    # Flush final pair.
    if current_prompt is not None and response_parts:
        pairs.append((prompt_index, current_prompt, " ".join(response_parts)))

    return pairs
