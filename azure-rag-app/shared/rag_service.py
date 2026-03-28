import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class RagService:
    """
    RAG orchestrator: embed question, retrieve top chunks, call LLM with context injection.
    """

    def __init__(self, embedder, search_index_client, llm_client, chat_model: str = "gpt-4o"):
        self.embedder = embedder
        self.search = search_index_client
        self.llm = llm_client
        self.chat_model = chat_model

    def answer_question(self, question: str, top_k: int = 5, topic: Optional[str] = None) -> Dict[str, Any]:
        if not question:
            raise ValueError("question required")

        # 1) embed
        q_emb = self.embedder.embed_text(question)

        # 2) vector search
        results = self.search.vector_search(query_embedding=q_emb, top_k=top_k, topic_filter=topic)

        # 3) build context
        context = "\n\n".join(f"[{r['id']}] {r['content']}" for r in results)

        system_prompt = (
            "You are a RAG assistant. Use ONLY the provided context to answer the question.\n"
            "If answer cannot be found in the context, say 'I don't know based on the indexed data.'"
        )

        user_prompt = f"Question: {question}\n\nContext:\n{context}"

        # 4) call LLM
        try:
            completion = self.llm.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=512,
                temperature=0.7,
            )

            answer = completion.choices[0].message.content
        except Exception as ex:
            logger.exception("LLM call failed: %s", ex)
            answer = "Error generating answer"

        # Build clean sources list with source_link and score (deduplicated)
        sources = []
        seen = set()
        for r in results:
            link = r.get("source_link")
            if not link:
                continue
            if link in seen:
                continue
            seen.add(link)
            sources.append({"source_link": link, "score": r.get("score"), "file_id": r.get("file_id")})

        return {"answer": answer, "sources": sources}

