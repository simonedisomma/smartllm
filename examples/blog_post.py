from smartllm import SmartLLM
from typing import Dict, Any, List
from pydantic import BaseModel, Field

openai_llm = SmartLLM("openai", "gpt-4")
anthropic_llm = SmartLLM("anthropic", "claude-3-sonnet-20240229")

class TopicList(BaseModel):
    topics: List[str] = Field(description="List of trending topics")

class BlogOutline(BaseModel):
    title: str = Field(description="Blog post title")
    sections: List[str] = Field(description="List of blog post sections")

class BlogSection(BaseModel):
    content: str = Field(description="Content of the blog section")

class TitleList(BaseModel):
    titles: List[str] = Field(description="List of catchy titles")

@openai_llm.configure("Generate a list of 5 trending topics for blog posts about {subject}")
def generate_topics(llm_response: TopicList, subject: str) -> List[str]:
    return llm_response.topics

@anthropic_llm.configure("Create an outline for a blog post about '{topic}'")
def create_outline(llm_response: BlogOutline, topic: str) -> Dict[str, Any]:
    return llm_response.model_dump()

@openai_llm.configure("Write a detailed section for '{section}' in the blog post about {topic}")
def write_section(llm_response: BlogSection, section: str, topic: str) -> str:
    return llm_response.content

@anthropic_llm.configure("Review and improve the following blog post section: {section}")
def review_section(llm_response: BlogSection, section: str) -> str:
    return llm_response.content

@openai_llm.configure("Generate 3 catchy titles for a blog post about {topic}")
def generate_titles(llm_response: TitleList, topic: str) -> List[str]:
    return llm_response.titles

def create_blog_post(subject: str):
    topics = openai_llm.generate_topics(subject=subject, response_format=TopicList)
    selected_topic = topics[0]
    
    outline = anthropic_llm.create_outline(topic=selected_topic, response_format=BlogOutline)
    
    sections = {}
    for section in outline["sections"]:
        content = openai_llm.write_section(section=section, topic=selected_topic, response_format=BlogSection)
        improved_content = anthropic_llm.review_section(section=content, response_format=BlogSection)
        sections[section] = improved_content
    
    titles = openai_llm.generate_titles(topic=selected_topic, response_format=TitleList)
    
    blog_post = {
        "title": titles[0],
        "topic": selected_topic,
        "outline": outline,
        "content": sections,
        "alternative_titles": titles[1:]
    }
    
    return blog_post

# Example usage
if __name__ == "__main__":
    blog_post = create_blog_post("Artificial Intelligence")
    print(blog_post)
