import json
import math
import re
import logging
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from app.services.query_analyzer import QueryAnalyzer
from app.services.task_planner import TaskPlanner
from app.services.rag_service import RAGService
from app.services.llm_service import LLMService


class STCAState(TypedDict, total=False):
    query: str
    doc_id: str
    user_id: str

    # Query Analysis
    intent: str
    keywords: List[str]
    time_sensitive: bool
    required_aspects: List[str]
    answer_style: str
    needs_cross_section_reasoning: bool

    # Task Plan
    steps: List[Any]
    retrieval_type: str
    top_k: int

    # Retrieval & Context
    retrieved_chunks: List[Dict[str, Any]]
    context_str: str
    context_similarity: float
    evidence_coverage: float
    dispersion: float
    agreement_score: float

    # LLM Output
    answer: str
    answer_alignment: float
    is_hybrid: bool

    # Confidence & Validation
    confidence_score: float
    is_supported: bool
    sources: List[str]
    reasoning: str

    # Advanced Filtering
    temporal_entities: Dict[str, List[str]]
    reasoning_type: str
    explanation: str
    matched_keywords: List[str]
    completeness_score: float


class STCAGraph:
    def __init__(self, llm_service: LLMService, rag_service: RAGService):
        self.llm_service = llm_service
        self.rag_service = rag_service

        self.query_analyzer = QueryAnalyzer(
            llm_client=llm_service.client,
            model=llm_service.model
        )
        self.task_planner = TaskPlanner(
            llm_client=llm_service.client,
            model=llm_service.model
        )

        builder = StateGraph(STCAState)

        builder.add_node("QueryAnalyzer", self.node_query_analyzer)
        builder.add_node("TaskPlanner", self.node_task_planner)
        builder.add_node("RetrievalAndContext", self.node_retrieval_and_context)
        builder.add_node("LLMGeneration", self.node_llm_generation)
        builder.add_node("ConfidenceScorer", self.node_confidence_scorer)
        builder.add_node("ValidationAgent", self.node_validation_agent)

        builder.set_entry_point("QueryAnalyzer")
        builder.add_edge("QueryAnalyzer", "TaskPlanner")
        builder.add_edge("TaskPlanner", "RetrievalAndContext")
        builder.add_edge("RetrievalAndContext", "LLMGeneration")
        builder.add_edge("LLMGeneration", "ConfidenceScorer")
        builder.add_edge("ConfidenceScorer", "ValidationAgent")
        builder.add_edge("ValidationAgent", END)

        self.graph = builder.compile()

    def node_query_analyzer(self, state: STCAState) -> Dict[str, Any]:
        analysis = self.query_analyzer.analyze_query(state["query"])
        return {
            "intent": analysis.get("intent", "search"),
            "keywords": analysis.get("keywords", [state["query"]]),
            "time_sensitive": analysis.get("time_sensitive", False),
            "required_aspects": analysis.get("required_aspects", []),
            "answer_style": analysis.get("answer_style", "extractive"),
            "needs_cross_section_reasoning": analysis.get("needs_cross_section_reasoning", False)
        }

    def node_task_planner(self, state: STCAState) -> Dict[str, Any]:
        analyzer_output = {
            "intent": state.get("intent", "search"),
            "keywords": state.get("keywords", [state["query"]]),
            "time_sensitive": state.get("time_sensitive", False),
            "required_aspects": state.get("required_aspects", []),
            "answer_style": state.get("answer_style", "extractive"),
            "needs_cross_section_reasoning": state.get("needs_cross_section_reasoning", False)
        }
        plan = self.task_planner.generate_plan(state["query"], analyzer_output)
        return {
            "steps": plan.get("steps", [{"query": state["query"], "aspect": "general"}]),
            "retrieval_type": plan.get("retrieval_type", "semantic"),
            "top_k": max(3, min(int(plan.get("top_k", 5)), 8))
        }

    async def node_retrieval_and_context(self, state: STCAState) -> Dict[str, Any]:
        all_chunks: List[Dict[str, Any]] = []
        hits_count = 0

        steps = state.get("steps", [state["query"]])
        top_k = state.get("top_k", 5)
        query_keywords = set(k.lower() for k in state.get("keywords", []))

        for sq in steps:
            q_text = sq.get("query", "") if isinstance(sq, dict) else str(sq)
            results = await self.rag_service.query_rag(
                doc_id=state["doc_id"],
                query=q_text,
                user_id=state["user_id"],
                return_scores=True,
                time_sensitive=state.get("time_sensitive", False),
                top_k=top_k
            )

            if results:
                hits_count += 1
                for res in results:
                    if not any(c.get("text") == res["text"] for c in all_chunks):
                        all_chunks.append(res)

        # True LLM list-wise reranking after vector search
        contradiction_risk = 0.0
        if all_chunks and len(all_chunks) > 1:
            rerank_prompt = f"""
            You are a Reranker and Contradiction Detector.
            Evaluate these chunks against the user's query: "{state['query']}"
            
            Chunks:
            """
            for idx, c in enumerate(all_chunks[:20]):
                rerank_prompt += f"[{idx}] {c['text'][:300]}\n"
            rerank_prompt += """
            Return ONLY a valid JSON object:
            {
               "ranked_indices": [list of chunk indices from most relevant to least relevant],
               "contradiction_detected": boolean,
               "contradiction_risk": 0.0
            }
            """
            try:
                res = self.llm_service.client.chat.completions.create(
                    messages=[{"role": "user", "content": rerank_prompt}],
                    model=self.llm_service.model,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                parsed = json.loads(res.choices[0].message.content)
                ranked = parsed.get("ranked_indices", [])
                contradiction_risk = float(parsed.get("contradiction_risk", 0.0))
                
                if ranked:
                    reranked_chunks = []
                    for idx in ranked:
                        if 0 <= idx < len(all_chunks):
                            reranked_chunks.append(all_chunks[idx])
                    
                    for idx, c in enumerate(all_chunks):
                        if idx not in ranked:
                            reranked_chunks.append(c)
                            
                    all_chunks = reranked_chunks
            except Exception as e:
                logging.warning(f"Reranking JSON parse error: {e}")

        if state.get("time_sensitive"):
            all_chunks = sorted(
                all_chunks[:15],
                key=lambda x: (
                    str(x.get("timestamp", "")),
                    x.get("confidence_score", 0.0)
                ),
                reverse=True
            )[:10]
        else:
            all_chunks = all_chunks[:10]

        formatted_contexts: List[str] = []
        semantic_scores: List[float] = []
        matched_keywords: set[str] = set()

        for i, ctx in enumerate(all_chunks):
            s_score = float(ctx.get("semantic_score", 0.0))
            semantic_scores.append(s_score)

            text = ctx.get("text", "")
            text_l = text.lower()
            for kw in query_keywords:
                if kw and kw in text_l:
                    matched_keywords.add(kw)

            timestamp = ctx.get("timestamp", "Unknown Time")
            display_date = (
                timestamp.split("T")[0]
                if isinstance(timestamp, str) and "T" in timestamp
                else str(timestamp).split(" ")[0]
            )

            formatted_contexts.append(
                f"[Source {i+1} | Date: {display_date} | Semantic: {s_score:.2f}]\n{text}"
            )

        context_str = "\n\n".join(formatted_contexts)

        context_similarity = (
            sum(semantic_scores) / len(semantic_scores)
            if semantic_scores else 0.0
        )

        evidence_coverage = (
            hits_count / len(steps)
            if steps else 0.0
        )

        dispersion = 0.0
        if len(semantic_scores) > 1:
            mean_score = sum(semantic_scores) / len(semantic_scores)
            variance = sum((s - mean_score) ** 2 for s in semantic_scores) / len(semantic_scores)
            dispersion = variance

        agreement_score = max(0.0, 1.0 - min(dispersion * 5.0, 1.0))

        sources_list: List[str] = []
        for c in all_chunks[:6]:
            text = c["text"].strip()
            if len(text) > 280:
                text = text[:277] + "..."
            if text and text not in sources_list:
                sources_list.append(text)

        return {
            "retrieved_chunks": all_chunks,
            "context_str": context_str,
            "context_similarity": context_similarity,
            "evidence_coverage": evidence_coverage,
            "dispersion": dispersion,
            "agreement_score": agreement_score,
            "sources": sources_list,
            "reasoning_type": "temporal_filtering",
            "matched_keywords": sorted(matched_keywords)
        }

    def _score_completeness(self, query: str, answer: str, required_aspects: List[str] = None) -> float:
        """
        Uses an LLM to grade completeness based on required aspects.
        Returns score between 0 and 1.
        """
        q = query.lower().strip()
        simple_factual = (
            q.startswith("what is ")
            or q.startswith("what are ")
            or q.startswith("define ")
            or q.startswith("who is ")
        )
        if simple_factual:
            return 1.0

        if not required_aspects:
            return 1.0

        prompt = f"""
        User asked: "{query}"
        Required aspects to cover: {required_aspects}
        Answer provided: "{answer}"
        
        Does the answer fully cover all the required aspects? 
        Return a JSON object with a single float field "completeness_score" from 0.0 (none) to 1.0 (all covered).
        """
        try:
            res = self.llm_service.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.llm_service.model,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            parsed = json.loads(res.choices[0].message.content)
            return float(parsed.get("completeness_score", 1.0))
        except Exception as e:
            logging.warning(f"Completeness JSON parse error: {e}")

        return 0.8

    def _estimate_answer_alignment(
        self,
        query: str,
        answer: str,
        sources: List[str],
        is_hybrid: bool,
        required_aspects: List[str] = None
    ) -> tuple[float, float]:
        if is_hybrid:
            return 0.35, 0.40

        ans = (answer or "").strip().lower()
        if not ans:
            return 0.10, 0.10

        negative_phrases = [
            "not found", "don't know", "not provided", "couldn't find", 
            "does not mention", "no information", "not mentioned",
            "cannot find", "unable to find", "isn't mentioned"
        ]
        if any(p in ans for p in negative_phrases):
            return 0.20, 0.25

        # Base alignment from answer richness
        if len(ans) < 20:
            base = 0.55
        elif len(ans) < 60:
            base = 0.78
        else:
            base = 0.90

        source_text = " ".join(sources).lower()
        overlap_hits = 0
        for token in set(ans.split()):
            if len(token) > 4 and token in source_text:
                overlap_hits += 1

        overlap_bonus = min(overlap_hits * 0.01, 0.05)
        answer_alignment = min(0.95, base + overlap_bonus)

        # NEW: LLM completeness score for multi-part questions
        completeness_score = self._score_completeness(query, answer, required_aspects=required_aspects)

        # If answer is incomplete, lower alignment too
        if completeness_score < 1.0:
            penalty = (1.0 - completeness_score) * 0.25
            answer_alignment = max(0.20, answer_alignment - penalty)

        return answer_alignment, completeness_score

    def node_llm_generation(self, state: STCAState) -> Dict[str, Any]:
        cs = state.get("context_similarity", 0.0)
        chunks_count = len(state.get("retrieved_chunks", []))
        is_hybrid = (chunks_count == 0) or (cs < 0.40)

        sources_override: List[str] = []
        temporal_entities: Dict[str, List[str]] = {}
        explanation = ""
        answer = ""
        
        required_aspects = state.get("required_aspects", [])
        aspect_instruction = ""
        if required_aspects:
            aspect_instruction = (
                f"\nImportant: The user asked for these aspects and your answer must cover ALL of them: "
                f"{', '.join(required_aspects)}."
            )

        if is_hybrid:
            prompt = f"""
You are a grounded assistant.

The uploaded document does not contain enough information to answer with confidence.

Instructions:
1. Start with exactly: "Not found in document."
2. Do not invent document facts or use general knowledge to answer.
3. Keep the answer concise.

Weak Context:
{state.get("context_str", "")}

User Question:
{state["query"]}
"""
            try:
                response = self.llm_service.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.llm_service.model,
                    temperature=0.2
                )
                answer = response.choices[0].message.content.strip()
                explanation = "Hybrid fallback triggered because retrieved document support was too weak."
            except Exception as e:
                logging.error(f"Hybrid generation error: {e}")
                answer = "Not found in document."
                explanation = "Hybrid fallback triggered but model generation failed."
        else:
            prompt = f"""
You are a precise document-grounded assistant.

Return ONLY valid JSON in this exact schema:
{{
  "answer": "Clear and complete answer in 1-2 sentences.",
  "explanation": "One sentence explaining why the answer is supported by the retrieved context.",
  "sources": ["Exact short supporting quote 1", "Exact short supporting quote 2"],
  "source_format": "Indicate 'image', 'table', 'text', or 'mixed' based on whether the supporting context came from a table, an image description, plain text, or a mix.",
  "temporal_entities": {{}}
}}

Rules:
1. Use ONLY the provided context.
2. If the answer is not present in the context, set the "answer" field to "I'm sorry, but I couldn't find any information about this in the provided document." and do not use your own knowledge.
3. The answer may summarize or combine multiple supporting lines.
4. Make the answer complete, not just keywords.
5. Keep sources short and exact.
6. If the user asked for comparison, use cases, risks, benefits, or best practices, cover every requested aspect.
7. Do not add anything outside the JSON.
{aspect_instruction}

Context:
{state.get("context_str", "")}

User Question:
{state["query"]}
"""
            try:
                response = self.llm_service.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.llm_service.model,
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                raw_out = response.choices[0].message.content.strip()
                parsed = json.loads(raw_out)

                answer = (parsed.get("answer") or "").strip()
                explanation = (parsed.get("explanation") or "").strip()
                temporal_entities = parsed.get("temporal_entities", {}) or {}
                sources_override = parsed.get("sources", []) or []
                source_format = parsed.get("source_format", "text")

                if not answer:
                    answer = "I could not generate a grounded answer from the document."
                if not explanation:
                    explanation = "The answer is based on the most relevant retrieved context."
            except Exception as e:
                answer = "I apologize, but I couldn't generate a grounded answer from the document."
                explanation = f"JSON generation/parsing error: {str(e)}"
                temporal_entities = {}
                source_format = "text"

        answer_alignment, completeness_score = self._estimate_answer_alignment(
            query=state["query"],
            answer=answer,
            sources=sources_override or state.get("sources", []),
            is_hybrid=is_hybrid,
            required_aspects=required_aspects
        )

        result = {
            "answer": answer,
            "answer_alignment": answer_alignment,
            "completeness_score": completeness_score,
            "is_hybrid": is_hybrid,
            "temporal_entities": temporal_entities,
            "source_format": source_format if not is_hybrid else "model knowledge",
            "explanation": explanation
        }

        if sources_override:
            result["sources"] = sources_override

        return result

    def node_confidence_scorer(self, state: STCAState) -> Dict[str, Any]:
        # Slightly adjusted weights
        a, b, c, d, e, f, g = 0.28, 0.16, 0.18, 0.14, 0.08, 0.06, 0.10

        s_topk = float(state.get("context_similarity", 0.0))
        coverage = float(state.get("evidence_coverage", 0.0))
        support = 1.0
        agreement = float(state.get("agreement_score", 0.0))
        dispersion = float(state.get("dispersion", 0.0))
        hallucination_risk = 1.0 if state.get("is_hybrid") else 0.10
        completeness = float(state.get("completeness_score", 1.0))

        raw_score = (
            (a * s_topk)
            + (b * coverage)
            + (c * support)
            + (d * agreement)
            + (g * completeness)
            - (e * dispersion)
            - (f * hallucination_risk)
        )

        scaled_score = raw_score * 3.0
        sigmoid = 1 / (1 + math.exp(-scaled_score))

        confidence = sigmoid
        explanation = state.get("explanation", "Completed execution via standard pipeline.")

        if state.get("is_hybrid"):
            confidence = min(confidence, 0.35)
            reasoning = (
                f"{explanation} | Hybrid cap applied "
                f"(S: {s_topk:.2f}, EC: {coverage:.2f}, AGR: {agreement:.2f})"
            )
        else:
            first_source = ""
            if state.get("sources"):
                first_source = state["sources"][0][:120]

            reasoning = (
                f"{explanation} | Scaled Conf: {confidence:.2f} "
                f"(S: {s_topk:.2f}, EC: {coverage:.2f}, AGR: {agreement:.2f}, "
                f"AA: {state.get('answer_alignment', 0.0):.2f}, "
                f"COMP: {completeness:.2f}, Disp: {dispersion:.2f})"
            )
            if first_source:
                reasoning += f" | Evidence: {first_source}"

        confidence_pct = round(max(0.0, min(1.0, confidence)) * 100, 2)

        return {
            "confidence_score": confidence_pct,
        "reasoning": reasoning
        }

    def node_validation_agent(self, state: STCAState) -> Dict[str, Any]:
        prompt = f"""
        Determine whether the answer is supported by the context and fully answers the user's question.
        
        Rules:
        1. The answer does NOT need to exactly match the wording of the context.
        2. Reply YES if it's a faithful summary grounded in context.
        3. Reply NO if it hallucinates or contradicts the document.
        4. (CRITICAL) The query required these aspects: {state.get("required_aspects", [])}
           If ANY of those aspects are missing from the answer, reply NO.
           
        Reply ONLY with YES or NO.
        
        Context:
        {state.get("context_str", "")[:2600]}
        
        Question:
        {state.get("query", "")}
        
        Answer:
        {state.get("answer", "")}
        """
        try:
            val_res = self.llm_service.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.llm_service.model,
                temperature=0.0
            )
            val_text = val_res.choices[0].message.content.strip().upper()
            is_supported = val_text.startswith("YES")
        except Exception as e:
            logging.warning(f"Validation agent error: {e}")
            is_supported = True

        final_conf = float(state.get("confidence_score", 0.0))
        reasoning = state.get("reasoning", "")
        completeness = float(state.get("completeness_score", 1.0))

        strong_grounding = (
            state.get("evidence_coverage", 0.0) >= 1.0
            and state.get("agreement_score", 0.0) >= 0.90
            and state.get("answer_alignment", 0.0) >= 0.85
            and completeness >= 0.95
            and not state.get("is_hybrid", False)
        )

        if not is_supported and strong_grounding:
            is_supported = True
            reasoning = reasoning + " | Validation override applied due to strong grounding."

        elif not is_supported:
            final_conf *= 0.75
            reasoning = reasoning + " | Validation agent uncertain; soft penalty applied."

        # NEW: hard completeness penalty for partial answers
        simple_factual = state.get("query", "").lower().strip().startswith(("what is ", "what are ", "define ", "who is "))
        if simple_factual:
            completeness = 1.0
            
        if completeness < 0.80 and not state.get("is_hybrid", False):
            final_conf *= 0.85
            reasoning = reasoning + " | Completeness penalty applied."

        return {
            "is_supported": is_supported,
            "confidence_score": round(final_conf, 2),
            "reasoning": reasoning
        }

    async def execute(self, query: str, doc_id: str, user_id: str) -> Dict[str, Any]:
        initial_state: STCAState = {
            "query": query,
            "doc_id": doc_id,
            "user_id": user_id
        }

        result = await self.graph.ainvoke(initial_state)

        final_output = {
            "answer": result.get("answer", ""),
            "confidence_score": result.get("confidence_score", 0.0),
            "sources": result.get("sources", []),
            "source_format": result.get("source_format", "text"),
            "reasoning": result.get("reasoning", ""),
            "reasoning_type": result.get("reasoning_type", "semantic_similarity"),
            "source_type": "document" if not result.get("is_hybrid") else "document + model knowledge",
            "is_supported": result.get("is_supported", True)
        }

        temps = result.get("temporal_entities")
        if temps and isinstance(temps, dict) and any(
            isinstance(v, list) and len(v) > 0 for v in temps.values()
        ):
            final_output["temporal_entities"] = temps

        if result.get("is_hybrid"):
            final_output["sources"] = ["No direct information found in uploaded document"]
            final_output["note"] = "Answer generated from model knowledge due to missing document context"

        return final_output