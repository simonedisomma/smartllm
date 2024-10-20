import unittest
from smartllm import SmartLLM
from smartllm.drivers import OpenAIDriver

class TestSmartLLM(unittest.TestCase):
    def setUp(self):
        self.openai_llm = SmartLLM(OpenAIDriver("gpt-4"))

    def test_llm_response(self):
        # Test that the LLM returns a non-empty string response
        prompt = "Hello, world!"
        response = self.openai_llm.generate(prompt)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_configure_decorator(self):
        # Test that the configure decorator works
        @self.openai_llm.configure("Greet {name}")
        def greet(llm_response: str, name: str) -> str:
            return f"LLM says: {llm_response}"

        result = greet("Alice")
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("LLM says:"))

    def test_error_handling(self):
        # Test error handling when an invalid prompt is provided
        with self.assertRaises(ValueError):
            self.openai_llm.generate("")

if __name__ == '__main__':
    unittest.main()


