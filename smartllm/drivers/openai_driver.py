from .base import LLMDriver
import openai
import os

class OpenAIDriver(LLMDriver):
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except openai.error.OpenAIError as e:
            print(f"OpenAI API error: {str(e)}")
            return f"Error: {str(e)}"
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return f"Unexpected error occurred: {str(e)}"
