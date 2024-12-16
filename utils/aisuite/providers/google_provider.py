from ..provider import Provider
import os
import openai

class GoogleProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Google provider with the given configuration.
        Pass the entire configuration dictionary to the GenerativeModel constructor.
        """
        config.setdefault("api_key", os.getenv("GEMINI_API_KEY"))
        if not config["api_key"]:
            raise ValueError("Google API key is missing. Please provide it in the config or set the GOOGLE_API_KEY environment variable.")
        
        # Pass the entire config to the GenerativeModel constructor
        self.client = openai.OpenAI(**config, base_url="https://generativelanguage.googleapis.com/v1beta")

    def chat_completions_create(self, model, messages, **kwargs):
        # Any exception raised by OpenAI will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs  # Pass any additional arguments to the OpenAI API
        )