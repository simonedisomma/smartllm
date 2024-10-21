from smartllm import SmartLLM
from smartllm.drivers import OpenAIDriver
from typing import List, Dict
import logging
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = SmartLLM(provider_id="openai", model_id="chatgpt-4o-latest")

class OutlineResponse(BaseModel):
    outline: List[str] = Field(description="List of outline points")

class SlideContent(BaseModel):
    content: str = Field(description="Content of the slide")

@llm.configure("Create a concise outline for a presentation about tokenizers in natural language processing. Limit to 5-7 main points.")
def create_outline(llm_response: OutlineResponse, **kwargs) -> List[str]:
    if isinstance(llm_response, OutlineResponse):
        return llm_response.outline[:7]  # Limit to 7 items max
    elif isinstance(llm_response, str):
        logger.warning(f"Received string response instead of OutlineResponse: {llm_response}")
        try:
            # Attempt to parse the string as JSON
            import json
            parsed = json.loads(llm_response)
            if isinstance(parsed, dict):
                for key in parsed.keys():
                    if 'outline' in key.lower() or 'presentation' in key.lower():
                        return parsed[key][:7]
        except json.JSONDecodeError:
            logger.error(f"Failed to parse string response as JSON: {llm_response}")
    logger.error(f"Unexpected response type in create_outline: {type(llm_response)}")
    return []

@llm.configure("Generate brief, informative content (2-3 sentences) for a slide titled '{slide_title}' about tokenizers.")
def generate_slide_content(llm_response: SlideContent, slide_title: str, **kwargs) -> str:
    if isinstance(llm_response, SlideContent):
        return llm_response.content.strip()
    elif isinstance(llm_response, str):
        logger.warning(f"Received string response instead of SlideContent: {llm_response}")
        try:
            # Attempt to parse the string as JSON
            import json
            parsed = json.loads(llm_response)
            if isinstance(parsed, dict):
                for key in parsed.keys():
                    if 'content' in key.lower():
                        return parsed[key].strip()
        except json.JSONDecodeError:
            logger.error(f"Failed to parse string response as JSON: {llm_response}")
    logger.error(f"Unexpected response type in generate_slide_content: {type(llm_response)}")
    return "Content generation failed. Please try again."

def create_presentation() -> List[Dict[str, str]]:
    try:
        outline = llm.create_outline(_caller='create_presentation', response_format=OutlineResponse)
        if not outline:
            logger.warning("Failed to create outline. Returning empty presentation.")
            return []

        slides = []
        for slide_title in outline:
            content = llm.generate_slide_content(slide_title=slide_title, _caller='create_presentation', response_format=SlideContent)
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
