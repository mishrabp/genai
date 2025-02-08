## This is a demo program to summerize youtube video
import os
import tiktoken
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI #AzureOpenAI


class LLM:
    _client = None  # Static variable to hold the API client

    @staticmethod
    def initialize():
        """Loads API key and initializes the OpenAI client."""
        load_dotenv()  # Load .env file if present
        api_key = os.getenv("AZURE_OPENAI_KEY")
        model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        # print(f"{api_key}-{model_name}-{azure_endpoint}-{api_version}")

        if not api_key:
            raise ValueError("Missing API Key! Set AZURE_OPENAI_KEY in environment.")

        LLM._client = AzureChatOpenAI(model_name=model_name,
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_endpoint)

    @staticmethod
    def get_client():
        """Returns the OpenAI client instance."""
        if LLM._client is None:
            LLM.initialize()
        return LLM._client

    @staticmethod
    def num_tokens_from_messages(messages):

        """
        Return the number of tokens used by a list of messages.
        Adapted from the Open AI cookbook token counter
        """

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        # Each message is sandwiched with <|start|>role and <|end|>
        # Hence, messages look like: <|start|>system or user or assistant{message}<|end|>

        tokens_per_message = 3 # token1:<|start|>, token2:system(or user or assistant), token3:<|end|>

        num_tokens = 0

        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))

        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

        return num_tokens