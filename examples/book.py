import os
import logging
import time
import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from smartllm import SmartLLM

# Set up logging
logging.basicConfig(level=logging.INFO)  # Changed to DEBUG level
logger = logging.getLogger(__name__)

openai_llm = SmartLLM(provider_id="openai", model_id="chatgpt-4o-latest")
anthropic_llm = SmartLLM(provider_id="anthropic", model_id="claude-3-sonnet-20240229")

class BookStructure(BaseModel):
    title: str = Field(description="Book title")
    chapters: List[Dict[str, Any]] = Field(description="List of chapters with their details", max_length=10)

class ChapterContent(BaseModel):
    content: str = Field(description="Content of the chapter, limited to one page")

class ChapterReview(BaseModel):
    review: str = Field(description="Brief review of the chapter")
    improvements: str = Field(description="Concise improvements to the chapter")
    rating: int = Field(description="Rating of the chapter (1-10)")

class StyleGuide(BaseModel):
    guide: str = Field(description="Style guide for the book")

class GlobalOutline(BaseModel):
    outline: str = Field(description="Global outline of the book")

class TerminologyGlossary(BaseModel):
    glossary: Dict[str, str] = Field(description="Terminology glossary for the book")

@openai_llm.configure("Create a detailed high-level structure for a book about {topic}. Include a title and a list of up to 10 chapters with their main points. Each chapter should be brief enough to fit on one page.")
def ideate_book_structure(llm_response: BookStructure, topic: str) -> Dict[str, Any]:
    logger.debug(f"Ideating book structure for topic: {topic}")
    return llm_response.model_dump()

@anthropic_llm.configure("Refine the following book structure, ensuring coherence and brevity. Limit to 10 chapters maximum, each fitting on one page: {structure}")
def improve_structure(llm_response: BookStructure, structure: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug("Improving book structure")
    return llm_response.model_dump()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
@openai_llm.configure("Write a one-page chapter for '{chapter}' in the book about {topic}, covering the following points: {points}. Follow the style guide: {style_guide}. Consider the global outline: {global_outline}. Reference the previous chapter if applicable: {previous_chapter}. Use terms from the glossary: {terminology_glossary}")
def write_chapter(llm_response: ChapterContent, chapter: str, topic: str, points: List[str], style_guide: str, global_outline: str, previous_chapter: str, terminology_glossary: Dict[str, str]) -> str:
    logger.debug(f"Writing chapter: {chapter}")
    return llm_response.content

@anthropic_llm.configure("Critically review the following one-page chapter as an experienced editor: {chapter}")
def review_chapter(llm_response: ChapterReview, chapter: str) -> Dict[str, Any]:
    logger.debug("Reviewing chapter")
    return llm_response.model_dump()

@openai_llm.configure("Rewrite the following chapter based on the review and improvements: {original_chapter}\n\nReview: {review}\nImprovements: {improvements}")
def rewrite_chapter(llm_response: ChapterContent, original_chapter: str, review: str, improvements: str) -> str:
    logger.debug("Rewriting chapter based on review")
    return llm_response.content

@openai_llm.configure("Provide a brief summary of the following book based on its chapters: {chapters}")
def summarize_book(llm_response: str, chapters: Dict[str, Dict[str, Any]]) -> str:
    logger.debug("Summarizing book")
    return llm_response.strip()

@openai_llm.configure("Create a style guide for a book about {topic} with the following structure: {structure}")
def create_style_guide(llm_response: StyleGuide, topic: str, structure: Dict[str, Any]) -> str:
    logger.debug(f"Creating style guide for topic: {topic}")
    return llm_response.guide

@openai_llm.configure("Initialize a global outline based on the following book structure: {structure}")
def initialize_global_outline(llm_response: GlobalOutline, structure: Dict[str, Any]) -> str:
    logger.debug("Initializing global outline")
    return llm_response.outline

@openai_llm.configure("Initialize a terminology glossary for a book about {topic}")
def initialize_terminology_glossary(llm_response: TerminologyGlossary, topic: str) -> Dict[str, str]:
    logger.debug(f"Initializing terminology glossary for topic: {topic}")
    return llm_response.glossary

@openai_llm.configure("Review and update the global outline considering the current chapter: {current_chapter}")
def review_global_outline(llm_response: GlobalOutline, global_outline: str, current_chapter: Dict[str, Any]) -> str:
    logger.debug("Reviewing global outline")
    return llm_response.outline

@openai_llm.configure("Update the global outline with the new chapter: {new_chapter}")
def update_global_outline(llm_response: GlobalOutline, global_outline: str, new_chapter: str) -> str:
    logger.debug("Updating global outline")
    return llm_response.outline

@openai_llm.configure("Update the terminology glossary with new terms from the chapter: {new_chapter}")
def update_terminology_glossary(llm_response: TerminologyGlossary, glossary: Dict[str, str], new_chapter: str) -> Dict[str, str]:
    logger.debug("Updating terminology glossary")
    return llm_response.glossary

@openai_llm.configure("Add inter-chapter references to the content based on the global outline: {global_outline}")
def add_inter_chapter_references(llm_response: ChapterContent, content: str, global_outline: str) -> str:
    logger.debug("Adding inter-chapter references")
    return llm_response.content

@openai_llm.configure("Perform a final consistency check on the book chapters: {chapters}")
def perform_final_consistency_check(llm_response: str, chapters: Dict[str, Dict[str, Any]], style_guide: str, global_outline: str, terminology_glossary: Dict[str, str]) -> str:
    logger.debug("Performing final consistency check")
    return llm_response.strip()

def create_book(topic: str) -> Dict[str, Any]:
    try:
        logger.debug(f"Creating book on topic: {topic}")
        
        structure = openai_llm.ideate_book_structure(topic=topic, response_format=BookStructure)
        logger.debug(f"Initial structure: {structure}")
        
        improved_structure = anthropic_llm.improve_structure(structure=structure, response_format=BookStructure)
        logger.debug(f"Improved structure: {improved_structure}")
        
        if not isinstance(improved_structure, dict) or 'chapters' not in improved_structure:
            logger.error(f"Invalid improved structure: {improved_structure}")
            raise ValueError("Invalid book structure returned")
        
        style_guide = openai_llm.create_style_guide(topic=topic, structure=improved_structure, response_format=StyleGuide)
        global_outline = openai_llm.initialize_global_outline(structure=improved_structure, response_format=GlobalOutline)
        terminology_glossary = openai_llm.initialize_terminology_glossary(topic=topic, response_format=TerminologyGlossary)
        
        chapters = {}
        previous_chapter = None
        for i, chapter in enumerate(improved_structure.get("chapters", [])[:10], 1):
            logger.debug(f"Processing chapter {i}: {chapter}")
            
            # Check if 'title' is in the chapter dictionary
            if 'title' not in chapter:
                logger.error(f"Chapter {i} is missing 'title': {chapter}")
                chapter['title'] = f"Chapter {i}"  # Assign a default title
            
            global_outline = openai_llm.review_global_outline(global_outline=global_outline, current_chapter=chapter, response_format=GlobalOutline)
            
            start_time = time.time()
            try:
                params = openai_llm.validate_params(
                    write_chapter,
                    chapter=chapter["title"],
                    topic=topic,
                    points=chapter.get("main_points", []),
                    style_guide=style_guide,
                    global_outline=global_outline,
                    previous_chapter=previous_chapter if previous_chapter else "",
                    terminology_glossary=terminology_glossary
                )
                content = openai_llm.write_chapter(
                    **params,
                    response_format=ChapterContent
                )
            except (TypeError, ValueError) as e:
                logger.error(f"Error in write_chapter: {str(e)}")
                logger.info("Falling back to simplified write_chapter call")
                content = openai_llm.write_chapter(
                    chapter=chapter["title"],
                    topic=topic,
                    points=chapter.get("main_points", []),
                    response_format=ChapterContent
                )
            end_time = time.time()
            logger.info(f"Chapter {i} written in {end_time - start_time:.2f} seconds")
            
            time.sleep(1)
            
            review = anthropic_llm.review_chapter(
                chapter=content,
                response_format=ChapterReview
            )
            rewritten_content = openai_llm.rewrite_chapter(
                original_chapter=content,
                review=review["review"],
                improvements=review["improvements"],
                response_format=ChapterContent
            )
            
            global_outline = openai_llm.update_global_outline(global_outline=global_outline, new_chapter=rewritten_content, response_format=GlobalOutline)
            terminology_glossary = openai_llm.update_terminology_glossary(glossary=terminology_glossary, new_chapter=rewritten_content, response_format=TerminologyGlossary)
            rewritten_content = openai_llm.add_inter_chapter_references(content=rewritten_content, global_outline=global_outline, response_format=ChapterContent)
            
            chapters[chapter["title"]] = {"content": rewritten_content, "review": review}
            previous_chapter = rewritten_content
            
            os.makedirs("book", exist_ok=True)
            with open(f"book/chapter_{i:02d}.md", "w") as f:
                f.write(f"# {chapter['title']}\n\n{rewritten_content}")
        
        summary = openai_llm.summarize_book(chapters=chapters)
        final_consistency_check = openai_llm.perform_final_consistency_check(
            chapters=chapters,
            style_guide=style_guide,
            global_outline=global_outline,
            terminology_glossary=terminology_glossary
        )
        
        logger.debug("Book creation completed")
        return {
            "structure": improved_structure,
            "chapters": chapters,
            "summary": summary,
            "style_guide": style_guide,
            "global_outline": global_outline,
            "terminology_glossary": terminology_glossary,
            "consistency_check": final_consistency_check
        }
    except Exception as e:
        logger.error(f"Error in create_book: {str(e)}", exc_info=True)
        return {"error": str(e)}

if __name__ == "__main__":
    openai_llm.clear_function_calls()
    anthropic_llm.clear_function_calls()
    book = create_book("Artificial Intelligence Ethics")
    if "error" not in book:
        print(f"Book structure: {book['structure']}")
        print(f"Number of chapters: {len(book['chapters'])}")
        print(f"Summary: {book['summary'][:100]}...")
        print("\nChapters saved in /book/ folder:")
        for filename in sorted(os.listdir("book")):
            print(f"  - {filename}")
    else:
        print(f"Error: {book['error']}")
    
    # Generate the flowchart
    openai_llm.generate_flowchart('book_creation_flowchart.png')
    anthropic_llm.generate_flowchart('book_review_flowchart.png')
