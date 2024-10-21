import logging
from typing import Dict, Any, List, Union
import json
from pydantic import BaseModel, Field
from smartllm import SmartLLM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = SmartLLM("openai", "chatgpt-4o-latest")

class BlogPost(BaseModel):
    title: str = Field(description="Blog post title")
    content: str = Field(description="Blog post content")

@llm.configure("Create a blog post about {topic} for business users. Include a catchy title and informative content.")
def create_blog_post(llm_response: Union[BlogPost, str], topic: str) -> Dict[str, Any]:
    logger.info(f"Creating blog post for topic: {topic}")
    if isinstance(llm_response, BlogPost):
        return llm_response.dict()
    elif isinstance(llm_response, str):
        try:
            blog_post = json.loads(llm_response)
            return blog_post
        except json.JSONDecodeError:
            logger.error(f"Failed to parse response into JSON: {llm_response}")
            return {"title": "Error", "content": llm_response}
    else:
        logger.error(f"Unexpected response type: {type(llm_response)}")
        return {"title": "Error", "content": "Unexpected response type"}

def generate_and_save_blog_post(topic: str):
    logger.info(f"Starting blog post creation for topic: {topic}")
    
    blog_post = llm.create_blog_post(topic=topic)
    
    if not blog_post or "title" not in blog_post or "content" not in blog_post:
        logger.error(f"Invalid blog post generated: {blog_post}")
        return None
    
    filename = f"blog_post_{blog_post['title'].replace(' ', '_')}.md"
    with open(filename, "w") as f:
        f.write(f"# {blog_post['title']}\n\n{blog_post['content']}")
    
    logger.info(f"Blog post created and saved to: {filename}")
    return filename

# Example usage
if __name__ == "__main__":
    filename = generate_and_save_blog_post("The impact of AI on drug discovery")
    if filename:
        print(f"Blog post saved to: {filename}")
    else:
        print("Failed to generate blog post")
