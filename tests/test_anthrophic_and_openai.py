import unittest
import json
from smartllm import SmartLLM
from pydantic import BaseModel, Field
from typing import Type, List

class TestAnthropicAndOpenAI(unittest.TestCase):
    def setUp(self):
        self.anthropic_llm = SmartLLM("anthropic", "claude-3-sonnet-20240229")
        self.openai_llm = SmartLLM("openai", "gpt-4o-mini")

    def test_anthropic_and_openai_interaction(self):
        class BlogOutline(BaseModel):
            title: str = Field(description="Blog post title")
            sections: List[str] = Field(description="List of blog post sections")

        class BlogSection(BaseModel):
            content: str = Field(description="Content of the blog section")

        def create_outline(topic: str) -> dict:
            prompt = f"Create an outline for an engaging blog post about '{topic}' tailored for business users"
            return self.anthropic_llm.generate(prompt=prompt, response_format=BlogOutline)

        def write_section(section: str, topic: str) -> str:
            prompt = f"Write a detailed section for '{section}' in the blog post about {topic}. Make it interesting and accessible for business users, with a touch of technical insight explained simply."
            response = self.openai_llm.generate(prompt=prompt, response_format=BlogSection)
            return response.content if isinstance(response, BlogSection) else response['content']

        def review_section(section: str) -> str:
            prompt = f"Review and improve the following blog post section: {section}"
            response = self.anthropic_llm.generate(prompt=prompt, response_format=BlogSection)
            return response.content if isinstance(response, BlogSection) else response['content']

        # Test the interaction between Anthropic and OpenAI
        topic = "AI in drug discovery"
        outline = create_outline(topic)
        
        self.assertIsInstance(outline, dict)
        self.assertIn("title", outline)
        self.assertIn("sections", outline)
        
        for section in outline["sections"]:
            content = write_section(section, topic)
            self.assertIsInstance(content, str)
            
            improved_content = review_section(content)
            self.assertIsInstance(improved_content, str)
            
            print(f"Section: {section}")
            print(f"Original content: {content[:100]}...")
            print(f"Improved content: {improved_content[:100]}...")
            print("\n")

        # Print the final outline for manual inspection
        print(json.dumps(outline, indent=2))
