from typing import Any, Type, Callable
import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableBinding
from langchain_core.exceptions import OutputParserException
from json_repair import repair_json

from app.helpers.pydantic import _ensure_llm_and_extra_body, to_strict_json_schema
from app.services.prompt_manager import get_prompt_template

MAX_ATTEMPTS_LLM = 2
LOGGER = logging.getLogger(__name__)
LLM_RETRY_PROMPT_KEY = "llm_retry"


class LLMHelper:
    async def safe_ainvoke(
            self,
            chain,
            callback,
            input_dict: dict,
            output_model,
            **kwargs: dict[str, Any],
    ):
        """
        Single-mode version: always uses OpenAI JSON schema when output_model is provided.
        If output_model is None -> return plain text.
        """
        attempt = 0
        response_text = ""
        parser: PydanticOutputParser | None = None

        while attempt <= MAX_ATTEMPTS_LLM:
            extra_body = None
            previous_response_format = None
            try:
                if output_model is not None:
                    _, extra_body = _ensure_llm_and_extra_body(chain)

                    output_model_dict = to_strict_json_schema(output_model)

                    response_format = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": output_model_dict["title"],
                            "strict": True,
                            "schema": output_model_dict,
                        },
                    }


                    previous_response_format = extra_body.get("response_format")
                    extra_body["response_format"] = response_format

                    parser = PydanticOutputParser(pydantic_object=output_model)

                    input_dict.setdefault("format_instructions", output_model_dict)

                LOGGER.debug(
                    "Invoking LLM chain attempt=%d model=%s", attempt + 1, getattr(chain, "llm", None)
                )
                response = await chain.ainvoke(input=input_dict, **kwargs, config={"callbacks": callback})

                response_text = response["text"]

                if not response_text:
                    raise OutputParserException("Empty response from LLM")

                if output_model is None:
                    return response_text

                parsed = await parser.aparse(self.validate_and_fix_json(response_text))
                return parsed

            except OutputParserException as e:
                attempt += 1
                LOGGER.warning("safe_ainvoke: output parse error: %s", e)

                if attempt > MAX_ATTEMPTS_LLM:
                    LOGGER.error("safe_ainvoke: Max retries reached.")
                    raise

                try:
                    await self.__append_llm_retry_prompt(
                        chain,
                        response_text,
                        str(e),
                        input_dict.get("format_instructions", ""),
                        attempt,
                    )
                except Exception as _e:
                    LOGGER.error("safe_ainvoke: error appending retry prompt: %s", _e)
                    raise

            except Exception as e:
                LOGGER.exception("safe_ainvoke: Error invoking LLM")
                raise
            finally:
                if extra_body is not None:
                    if previous_response_format is None:
                        extra_body.pop("response_format", None)
                    else:
                        extra_body["response_format"] = previous_response_format

    @classmethod
    def validate_and_fix_json(cls, response: str):
        invalid_json = '""'
        repair_attempt = repair_json(response, ensure_ascii=False).strip()
        return (
            repair_attempt
            if repair_attempt and repair_attempt != invalid_json
            else response
        )



    async def __append_llm_retry_prompt(
            self,
            chain,
            response: str,
            error_message: str,
            format_instructions: str,
            attempt: int,
    ) -> None:
        """Add llm_retry as a system message to given chain."""
        prompt_template = get_prompt_template(LLM_RETRY_PROMPT_KEY)

        error_message = error_message.replace(response, "")
        llm_retry = SystemMessagePromptTemplate.from_template(
            prompt_template,
            partial_variables={
                "previous_response": response,
                "error_message": error_message,
                "format_instructions": format_instructions,
            },
        )

        if isinstance(chain, RunnableBinding):
            chain.bound.name = f"{chain.bound.name} - Retry {attempt}"
            if isinstance(chain.prompt, ChatPromptTemplate):
                chain.prompt.messages.insert(-1, llm_retry)
            elif isinstance(chain.prompt, PromptTemplate):
                chain.prompt = ChatPromptTemplate(
                    messages=[
                        SystemMessagePromptTemplate.from_template(
                            chain.prompt.template
                        ),
                        llm_retry,
                    ],
                )
        else:
            msg = f"Unknown chain type {type(chain)}"
            raise ValueError(msg)
