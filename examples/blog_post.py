import os
import logging
from typing import Dict, Any, List, Union
from pydantic import BaseModel, Field
from smartllm import SmartLLM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai_llm = SmartLLM("openai", "chatgpt-4o-latest")
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

class StyleGuide(BaseModel):
    guidelines: str = Field(description="Style guidelines for the blog post")

@openai_llm.configure("Generate a list of 5 trending topics for blog posts about {subject} that would interest business users.")
def generate_topics(llm_response: TopicList, subject: str) -> List[str]:
    logger.info(f"Generating topics for subject: {subject}")
    logger.info(f"Generated topics: {llm_response.topics}")
    return llm_response.topics

@anthropic_llm.configure("Create an outline for an engaging blog post about '{topic}' tailored for business users")
def create_outline(llm_response: BlogOutline, topic: str) -> Dict[str, Any]:
    logger.info(f"Creating outline for topic: {topic}")
    return llm_response.dict()

@openai_llm.configure("Write a detailed section for '{section}' in the blog post about {topic}. Make it interesting and accessible for business users, with a touch of technical insight explained simply.")
def write_section(llm_response: BlogSection, section: str, topic: str, style_guide: str) -> str:
    logger.info(f"Writing section: {section}")
    return llm_response.content

@anthropic_llm.configure("Review and improve the following blog post section, ensuring it aligns with our style guide: {section}")
def review_section(llm_response: BlogSection, section: str, style_guide: str) -> str:
    logger.info(f"Reviewing section: {section}")
    return llm_response.content

@openai_llm.configure("Generate 3 catchy, business-oriented titles for a blog post about {topic}")
def generate_titles(llm_response: TitleList, topic: str) -> List[str]:
    logger.info(f"Generating titles for topic: {topic}")
    return llm_response.titles

@anthropic_llm.configure("Create a style guide for our blog post. It should be interesting, tailored for business users, with some technical elements explained simply. Avoid buzzwords but maintain a powerful tone.")
def create_style_guide(llm_response: Union[StyleGuide, str]) -> str:
    logger.info("Creating style guide")
    if isinstance(llm_response, StyleGuide):
        return llm_response.guidelines
    elif isinstance(llm_response, str):
        return llm_response
    else:
        logger.error(f"Unexpected response type: {type(llm_response)}")
        return ""

def create_blog_post(subject: str):
    logger.info(f"Starting blog post creation for subject: {subject}")
    style_guide = anthropic_llm.create_style_guide()
    
    topics = openai_llm.generate_topics(subject=subject)
    logger.info(f"Generated topics: {topics}")
    
    if not topics:
        logger.error("No topics generated")
        return None
    
    selected_topic = topics[0]
    logger.info(f"Selected topic: {selected_topic}")
    
    outline = anthropic_llm.create_outline(topic=selected_topic)
    logger.info(f"Generated outline: {outline}")
    
    if not outline or "sections" not in outline:
        logger.error(f"Invalid outline generated: {outline}")
        return None
    
    sections = {}
    for section in outline["sections"]:
        logger.info(f"Processing section: {section}")
        content = openai_llm.write_section(section=section, topic=selected_topic, style_guide=style_guide)
        improved_content = anthropic_llm.review_section(section=content, style_guide=style_guide)
        sections[section] = improved_content
    
    titles = openai_llm.generate_titles(topic=selected_topic)
    
    blog_post = {
        "title": titles[0] if titles else "Untitled",
        "topic": selected_topic,
        "outline": outline,
        "content": sections,
        "alternative_titles": titles[1:] if len(titles) > 1 else []
    }
    
    # Final review and improvement
    final_review_prompt = f"Review and improve this blog post to ensure it's engaging, informative, and aligns with our style guide: {blog_post}"
    final_improved_post = anthropic_llm.generate(final_review_prompt)
    
    # Save the blog post
    os.makedirs("blog_post", exist_ok=True)
    filename = f"blog_post/{blog_post['title'].replace(' ', '_')}.md"
    with open(filename, "w") as f:
        f.write(final_improved_post)
    
    logger.info(f"Blog post created and saved to: {filename}")
    return final_improved_post

# Example usage
if __name__ == "__main__":
    blog_post = create_blog_post("The impact of AI on the future of drug discovery eg. with Alfafold")
    if blog_post:
        print("Blog post created successfully.")
    else:
        print("Failed to create blog post.")
