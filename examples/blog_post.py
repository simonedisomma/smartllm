from smartllm import SmartLLM
from typing import Dict, Any, List

openai_llm = SmartLLM("openai", "gpt-4")
anthropic_llm = SmartLLM("anthropic", "claude-3-sonnet-20240229")

@openai_llm.configure("Generate a list of 5 trending topics for blog posts about {subject}")
def generate_topics(llm_response: str, subject: str) -> List[str]:
    # Process the LLM response
    topics = llm_response.split('\n')
    return [topic.strip() for topic in topics if topic.strip()]

@anthropic_llm.configure("Create an outline for a blog post about '{topic}'")
def create_outline(llm_response: str, topic: str) -> Dict[str, Any]:
    # Process the LLM response
    # For demonstration, we'll use a simple outline structure
    return {
        "title": topic,
        "sections": ["Introduction", "Main Point 1", "Main Point 2", "Main Point 3", "Conclusion"]
    }

@openai_llm.configure("Write a detailed section for '{section}' in the blog post about {topic}")
def write_section(llm_response: str, section: str, topic: str) -> str:
    # Process the LLM response
    return llm_response.strip()

@anthropic_llm.configure("Review and improve the following blog post section: {section}")
def review_section(llm_response: str, section: str) -> str:
    # Process the LLM response
    return llm_response.strip()

@openai_llm.configure("Generate 3 catchy titles for a blog post about {topic}")
def generate_titles(llm_response: str, topic: str) -> List[str]:
    # Process the LLM response
    titles = llm_response.split('\n')
    return [title.strip() for title in titles if title.strip()]

def create_blog_post(subject: str):
    # Generate topics
    topics = generate_topics(subject)
    
    # Select a topic (for this example, we'll use the first one)
    selected_topic = topics[0]
    
    # Create outline
    outline = create_outline(selected_topic)
    
    # Write sections
    sections = {}
    for section in outline["sections"]:
        content = write_section(section, selected_topic)
        improved_content = review_section(content)
        sections[section] = improved_content
    
    # Generate titles
    titles = generate_titles(selected_topic)
    
    # Compile the blog post
    blog_post = {
        "title": titles[0],  # Choose the first title
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
