from smartllm import SmartLLM
from smartllm.drivers import OpenAIDriver
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = SmartLLM(OpenAIDriver("gpt-4"))

@llm.configure("Create a concise outline for a presentation about tokenizers in natural language processing. Limit to 5-7 main points.")
def create_outline(llm_response: str) -> List[str]:
    try:
        outline = llm_response.split('\n')
        return [item.strip() for item in outline if item.strip()][:7]  # Limit to 7 items max
    except Exception as e:
        logger.error(f"Error in create_outline: {str(e)}")
        return []

@llm.configure("Generate brief, informative content (2-3 sentences) for a slide titled '{slide_title}' about tokenizers")
def generate_slide_content(llm_response: str, slide_title: str) -> str:
    try:
        return llm_response.strip()
    except Exception as e:
        logger.error(f"Error in generate_slide_content for '{slide_title}': {str(e)}")
        return "Content generation failed. Please try again."

def create_presentation() -> List[Dict[str, str]]:
    try:
        outline = llm.create_outline(_caller='create_presentation')
        if not outline:
            logger.warning("Failed to create outline. Returning empty presentation.")
            return []

        slides = []
        for slide_title in outline:
            content = llm.generate_slide_content(slide_title, _caller='create_presentation')
            slides.append({"title": slide_title, "content": content})
        return slides
    except Exception as e:
        logger.error(f"Error in create_presentation: {str(e)}")
        return []

if __name__ == "__main__":
    llm.clear_function_calls()  # Clear any previous function calls
    presentation = create_presentation()
    if not presentation:
        logger.error("Failed to create presentation.")
    else:
        for i, slide in enumerate(presentation, 1):
            print(f"Slide {i}:")
            print(f"Title: {slide['title']}")
            print(f"Content: {slide['content']}\n")
    
    # Generate the flowchart
    llm.generate_flowchart('presentation_flowchart.png')
