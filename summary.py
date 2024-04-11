import nltk
from typing import Any, List, Mapping, Optional
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer  # Import LuhnSummarizer

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM


nltk.download('punkt')

class CustomLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.summarize_text(prompt)

    @staticmethod
    def summarize_text(text: str) -> str:
        # Use LuhnSummarizer for summarization
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer_luhn = LuhnSummarizer()
        summary_1 = summarizer_luhn(parser.document, 2)
        dp = []
        for i in summary_1:
            lp = str(i)
            dp.append(lp)
        final_sentence = ' '.join(dp)
        return final_sentence

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"summarizer_type": "luhn"}  # Indicate LuhnSummarizer as the summarizer type

# Create an instance of CustomLLM
llm = CustomLLM()

paragraph = input("Enter the paragraph to summarize:\n")

# Invoke the instance with the provided paragraph using LuhnSummarizer
summary_luhn = llm.invoke(paragraph)
print("Summary:")
print(summary_luhn)
