import unittest
import os
from smartllm import SmartLLM

class TestSmartLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure that the OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set in the environment")
        cls.openai_llm = SmartLLM(provider_id="openai", model_id="gpt-3.5-turbo")

    def test_llm_response(self):
        prompt = "Say 'Hello, world!' in French."
        response = self.openai_llm.generate(prompt)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        self.assertIn("Bonjour", response)

    def test_configure_decorator(self):
        @self.openai_llm.configure("Greet {name} in Italian")
        def greet(llm_response: str, name: str) -> str:
            return f"LLM says: {llm_response}"

        result = greet(name="Alice")
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("LLM says:"))
        self.assertIn("Ciao", result)

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            self.openai_llm.generate("")

if __name__ == '__main__':
    unittest.main()




