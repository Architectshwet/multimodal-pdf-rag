PAGE_DESCRIPTION_PROMPT = (
    "You are a document transcription and visual description engine. "
    "Given a single PDF page image, extract all visible content top-to-bottom with maximum fidelity.\n\n"
    "Rules:\n"
    "1) Transcribe all readable text verbatim (headings, paragraphs, bullets, equations, units, symbols).\n"
    "2) Preserve structure: titles, sections, numbering, columns, and reading order.\n"
    "3) For tables: reproduce all headers and cell values exactly; then add a brief semantic description.\n"
    "4) For figures/diagrams/graphs: do NOT invent text; describe only what is visually shown and how it relates to nearby text.\n"
    "5) If any text or symbol is unclear or unreadable, explicitly mark it as [UNCLEAR].\n\n"
    "Output only factual content suitable for semantic retrieval. Do not summarize or paraphrase text."
)

QUERY_REWRITE_SYSTEM_PROMPT = (
    "Rewrite the latest user question into one standalone retrievable query. "
    "Use only information present in the conversation history and latest question. "
    "Do not add assumptions, explanations, or extra constraints. "
    "Return only the rewritten query text."
)

QA_SYSTEM_PROMPT = (
    "You are a multimodal PDF RAG assistant.\n"
    "Answer only from retrieved page images.\n"
    "Keep the answer within 3-4 short lines.\n"
    "If evidence is missing, reply exactly: There is not enough information in the retrieved images."
)


def build_query_rewrite_user_prompt(history_text: str, latest_question: str) -> str:
    return (
        f"Conversation history:\n{history_text or '(no history)'}\n\n"
        f"Latest question:\n{latest_question}\n\n"
        "Return one standalone retrievable query only."
    )


def build_qa_user_prompt(question: str, rewritten_query: str | None = None) -> str:
    cleaned_question = question.strip()
    cleaned_rewritten = (rewritten_query or "").strip()

    if not cleaned_rewritten or cleaned_rewritten.casefold() == cleaned_question.casefold():
        return (
            f"User question:\n{cleaned_question}\n\n"
            "Answer from attached retrieved PDF page images."
        )

    return (
        f"User question:\n{cleaned_question}\n\n"
        f"Rewritten retrieval query:\n{cleaned_rewritten}\n\n"
        "Answer from attached retrieved PDF page images."
    )
