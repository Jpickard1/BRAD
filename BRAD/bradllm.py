"""
This file constructs a langchain compatible llm from an instance of BRAD
"""
from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

class BradLLM(LLM):
    bot: Optional[Any] = None

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """

        if stop is not None:
            print("stop kwargs are not permitted.")
        return self.bot.invoke(prompt)
    
    @property
    def _llm_type(self) -> str:
        """
        This function allows BRAD to be inserted as an LLM into a LangChain or LangGraph
        """

        return "BRAD"
