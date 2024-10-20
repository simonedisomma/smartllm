from smartllm import SmartLLM
from smartllm.drivers import OpenAIDriver, AnthropicDriver
from typing import Dict, Any, List
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai_llm = SmartLLM(OpenAIDriver("gpt-4"))
anthropic_llm = SmartLLM(AnthropicDriver("claude-3-sonnet-20240229"))

@openai_llm.configure("Create a detailed high-level structure for a book about {topic}. Return the structure as a JSON string.")
def ideate_book_structure(llm_response: str, topic: str) -> Dict[str, Any]:
    logger.info(f"Ideating book structure for topic: {topic}")
    try:
        structure = json.loads(llm_response)
        logger.debug(f"Generated book structure: {structure}")
        return structure
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON")
        return {"error": "Failed to generate book structure"}

@anthropic_llm.configure("Improve and expand on the following book structure, adding depth and coherence: {structure}. Return the improved structure as a JSON string.")
def improve_structure(llm_response: str, structure: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Improving book structure")
    try:
        improved_structure = json.loads(llm_response)
        logger.debug(f"Improved book structure: {improved_structure}")
        return improved_structure
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON")
        return structure  # Return original structure if parsing fails

@openai_llm.configure("Create a detailed index for the following book structure: {structure}. Return the index as a JSON string with page numbers.")
def create_index(llm_response: str, structure: Dict[str, Any]) -> Dict[str, List[str]]:
    logger.info("Creating book index")
    try:
        index = json.loads(llm_response)
        logger.debug(f"Generated index: {index}")
        return index
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON")
        return {"index": []}

@openai_llm.configure("Write a detailed chapter for '{chapter}' in the book about {topic}, covering the following sections: {sections}. Return the content as a JSON string with sections as keys.")
def write_chapter(llm_response: str, chapter: str, topic: str, sections: List[str]) -> Dict[str, str]:
    logger.info(f"Writing chapter: {chapter}")
    try:
        content = json.loads(llm_response)
        logger.debug(f"Generated content for chapter {chapter}")
        return content
    except json.JSONDecodeError:
        logger.error(f"Failed to parse LLM response as JSON for chapter: {chapter}")
        return {"error": f"Failed to generate content for {chapter}"}

@anthropic_llm.configure("Critically review the following chapter as an experienced editor: {chapter}. Return the review as a JSON string with 'review' and 'rating' keys.")
def review_chapter(llm_response: str, chapter: Dict[str, str]) -> Dict[str, Any]:
    logger.info("Reviewing chapter")
    try:
        review = json.loads(llm_response)
        logger.debug(f"Chapter review: {review}")
        return review
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON")
        return {"review": "Error in review generation", "rating": 0}

@openai_llm.configure("Provide a comprehensive summary of the following book based on its chapters: {chapters}. Return the summary as a string.")
def summarize_book(llm_response: str, chapters: Dict[str, Dict[str, Any]]) -> str:
    logger.info("Summarizing book")
    summary = llm_response.strip()
    logger.debug(f"Generated book summary: {summary[:50]}...")
    return summary

def create_book(topic: str) -> Dict[str, Any]:
    logger.info(f"Creating book on topic: {topic}")
    
    structure = ideate_book_structure(topic)
    if "error" in structure:
        return {"error": "Failed to create book structure"}
    
    improved_structure = improve_structure(structure)
    
    index = create_index(improved_structure)
    
    chapters = {}
    for chapter in improved_structure.get("chapters", []):
        content = write_chapter(chapter["title"], topic, chapter.get("sections", []))
        if "error" not in content:
            review = review_chapter(content)
            chapters[chapter["title"]] = {"content": content, "review": review}
        else:
            logger.error(f"Skipping chapter '{chapter['title']}' due to content generation error")
    
    if not chapters:
        return {"error": "Failed to generate any chapters"}
    
    summary = summarize_book(chapters)
    
    logger.info("Book creation completed")
    return {
        "structure": improved_structure,
        "index": index,
        "chapters": chapters,
        "summary": summary
    }
