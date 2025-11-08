from typing import Any, List

from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.base import BasePromptTemplate
from app.core.context_vars.context_vars import FlowContextManager
from app.schema.common import NewsList, NewsIntentResponse
from app.services.langfuse_client import LangfuseCallbackHandler
from app.services.llm_helper.llm_helper import LLMHelper
from app.services.vector_reader.vector_reader import VectorStoreReader
from app.services.llm_provider import LLMProvider
from app.services.prompt_manager import get_prompt_template

BATCH_SIZE = 10
NEWS_PROMPT_KEY = "chat_response"
GENERAL_PROMPT_KEY = "general_response"
NEWS_INTENT_PROMPT_KEY = "news_intent"


def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


class ChatService(LLMHelper):
    def __init__(self, reader: VectorStoreReader):
        self.reader = reader
        self.llm = LLMProvider.get()
        self.prompt_name = None
        self.msg_id = FlowContextManager.get_user_message_id()
        self.conv_id = FlowContextManager.get_user_message_id()
        self.output_model = None

    async def handle_message(self, user_text: str) -> str:
        FlowContextManager.init_for_message()

        is_news = await self._detect_news_intent(user_text)
        prompt_key = NEWS_PROMPT_KEY if is_news else GENERAL_PROMPT_KEY
        prompt = self._load_prompt(prompt_key)
        input_payload = {"question": user_text}

        if is_news:
            docs = self.reader.retrieve(user_text)
            context = " ".join(d.page_content for d in docs) if docs else "No context."
            input_payload["context"] = context

        self.prompt_name = prompt_key

        handler = LangfuseCallbackHandler(
            session_id=str(self.conv_id),
            trace_name=self.prompt_name or "chat",
            metadata={"user_message_id": str(self.msg_id), "prompt_key": prompt_key},
        )
        chain = (
            LLMChain(
                name=self.prompt_name,
                llm=self.llm,
                prompt=prompt,
                callbacks=[handler],
            )
            .with_config(
                {
                    "langfuse_session_id": str(self.conv_id),
                    "metadata": {"user_message_id": str(self.msg_id), "prompt_key": prompt_key},
                }
            )
        )

        result = await self.safe_ainvoke(
            chain=chain,
            callback=[handler],
            input_dict=input_payload,
            output_model=self.output_model,
        )
        return result

    async def generate_news_summaries(self, data) -> NewsList:
        """
        Takes a NewsAPI-like payload:
        {
            "status": "ok",
            "totalResults": ...,
            "articles": [ ... ]
        }
        Injects list of articles into prompt, asks LLM to produce:
            NewsList(items=[NewsItem(...), ...])
        and returns that Pydantic model.
        """

        prompt_key = "news_summary"
        prompt = self._load_prompt(prompt_key)
        self.prompt_name = prompt_key
        handler = LangfuseCallbackHandler(
            session_id=str(self.conv_id),
            trace_name=self.prompt_name or "chat",
            metadata={"user_message_id": str(self.msg_id), "prompt_key": prompt_key},
        )

        chain = LLMChain(prompt=prompt, llm=self.llm, name=self.prompt_name, callbacks=[handler])

        result = await self.safe_ainvoke(
            chain=chain,
            callback=[handler],
            input_dict={"articles": data},
            output_model=NewsList,
        )

        return result

    async def _detect_news_intent(self, user_text: str) -> bool:
        prompt = self._load_prompt(NEWS_INTENT_PROMPT_KEY)
        handler = LangfuseCallbackHandler(
            session_id=str(self.conv_id),
            trace_name=NEWS_INTENT_PROMPT_KEY,
            metadata={"user_message_id": str(self.msg_id), "classification": True},
        )
        chain = (
            LLMChain(
                name=NEWS_INTENT_PROMPT_KEY,
                llm=self.llm,

                prompt=prompt,
                callbacks=[handler],
            )
            .with_config(
                {
                    "langfuse_session_id": str(self.conv_id),
                    "metadata": {"user_message_id": str(self.msg_id), "classification": True},
                }
            )
        )

        response = await self.safe_ainvoke(
            chain=chain,
            callback=[handler],
            output_model=NewsIntentResponse,
            input_dict={"question": user_text},
        )

        return bool(getattr(response, "is_news", False))

    @staticmethod
    def _load_prompt(prompt_key: str, *, prefer_langfuse: bool = True) -> BasePromptTemplate:
        template = get_prompt_template(prompt_key, prefer_langfuse=prefer_langfuse)
        if isinstance(template, BasePromptTemplate):
            return template
        if isinstance(template, str):
            structured = ChatService._build_chat_prompt_from_string(template)
            if structured:
                return structured
            return ChatPromptTemplate.from_template(template)
        msg = f"Unsupported prompt template type for key '{prompt_key}': {type(template)}"
        raise TypeError(msg)

    @staticmethod
    def _build_chat_prompt_from_string(template: str) -> BasePromptTemplate | None:
        stripped = template.strip()
        if not stripped:
            return None

        system_lines: List[str] = []
        human_lines: List[str] = []
        current_target = system_lines

        for line in stripped.splitlines():
            if "{" in line and "}" in line:
                current_target = human_lines
            current_target.append(line)

        system_text = "\n".join(system_lines).strip()
        human_text = "\n".join(human_lines).strip()

        if not system_text or not human_text:
            return None

        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_text),
                HumanMessagePromptTemplate.from_template(human_text),
            ]
        )
