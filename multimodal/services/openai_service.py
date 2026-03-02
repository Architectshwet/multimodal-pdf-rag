import base64
import mimetypes
from pathlib import Path

from openai import AsyncOpenAI

from multimodal.config import config
from multimodal.prompts.multimodal_rag_prompt import (
    PAGE_DESCRIPTION_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
    build_qa_user_prompt,
    build_query_rewrite_user_prompt,
)


class OpenAIService:
    def __init__(self) -> None:
        if not config.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        self.client = AsyncOpenAI(api_key=config.openai_api_key)

    @staticmethod
    def _file_to_data_url(image_path: str | Path) -> str:
        path = Path(image_path)
        mime, _ = mimetypes.guess_type(path.name)
        if not mime:
            mime = "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _extract_usage_tokens(response) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    async def describe_pdf_page(self, image_path: str | Path, page_number: int) -> tuple[str, dict[str, int]]:
        response = await self.client.chat.completions.create(
            model=config.vision_model,
            messages=[
                {
                    "role": "system",
                    "content": PAGE_DESCRIPTION_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Transcribe and describe PDF page {page_number} using the system rules.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._file_to_data_url(image_path),
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_completion_tokens=2200,
            temperature=0,
        )
        description = (response.choices[0].message.content or "").strip()
        token_usage = self._extract_usage_tokens(response)
        return description, token_usage

    async def rewrite_query(self, latest_question: str, history: list[dict]) -> str:
        history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history)

        response = await self.client.chat.completions.create(
            model=config.rewrite_model,
            messages=[
                {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_query_rewrite_user_prompt(
                        history_text=history_text,
                        latest_question=latest_question,
                    ),
                },
            ],
            max_completion_tokens=120,
            temperature=0,
        )

        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten or latest_question

    async def stream_answer_from_images(
        self,
        question: str,
        rewritten_query: str | None,
        image_paths: list[str],
    ):
        normalized_question = question.strip()
        normalized_rewritten = (rewritten_query or "").strip()
        qa_rewritten_query = (
            normalized_rewritten
            if normalized_rewritten and normalized_rewritten.casefold() != normalized_question.casefold()
            else None
        )

        user_content: list[dict] = [
            {
                "type": "text",
                "text": build_qa_user_prompt(
                    question=question,
                    rewritten_query=qa_rewritten_query,
                ),
            }
        ]

        for image_path in image_paths:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self._file_to_data_url(image_path),
                        "detail": "high",
                    },
                }
            )

        stream = await self.client.chat.completions.create(
            model=config.chat_model,
            messages=[
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=220,
            temperature=0.2,
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
