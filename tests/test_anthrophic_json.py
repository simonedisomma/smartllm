import unittest
import json
from smartllm import SmartLLM
from pydantic import BaseModel, Field
from typing import Type

class TestAnthropicJSON(unittest.TestCase):
    def setUp(self):
        self.anthropic_llm = SmartLLM("anthropic", "claude-3-sonnet-20240229")

    def test_anthropic_json_response(self):
        class BlogOutline(BaseModel):
            title: str = Field(description="Blog post title")
            sections: list[str] = Field(description="List of blog post sections")

        def create_outline(topic: str) -> dict:
            prompt = f"Create an outline for an engaging blog post about '{topic}' tailored for business users"
            return self.anthropic_llm.generate(prompt=prompt, response_format=BlogOutline)

        # Test the function with real API call
        result = create_outline(topic="AI in drug discovery")

        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn("title", result)
        self.assertIsInstance(result["title"], str)
        self.assertIn("sections", result)
        self.assertIsInstance(result["sections"], list)
        self.assertTrue(len(result["sections"]) > 0)

        # Print the result for manual inspection
        print(json.dumps(result, indent=2))
