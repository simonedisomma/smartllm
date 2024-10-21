import os
from typing import Dict, Any, List
import logging
from pydantic import BaseModel, Field
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

@openai_llm.configure("Create a detailed high-level structure for a book about {topic}. Include a title and a list of up to 10 chapters with their main points. Each chapter should be brief enough to fit on one page.")
def ideate_book_structure(llm_response: BookStructure, topic: str) -> Dict[str, Any]:
    logger.debug(f"Ideating book structure for topic: {topic}")
    return llm_response.model_dump()

@anthropic_llm.configure("Refine the following book structure, ensuring coherence and brevity. Limit to 10 chapters maximum, each fitting on one page: {structure}")
def improve_structure(llm_response: BookStructure, structure: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug("Improving book structure")
    return llm_response.model_dump()

@openai_llm.configure("Write a one-page chapter for '{chapter}' in the book about {topic}, covering the following points: {points}")
def write_chapter(llm_response: ChapterContent, chapter: str, topic: str, points: List[str]) -> str:
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

def create_book(topic: str) -> Dict[str, Any]:
    try:
        logger.debug(f"Creating book on topic: {topic}")
        
        structure = openai_llm.ideate_book_structure(topic=topic, _caller='create_book', response_format=BookStructure)
        logger.debug(f"Initial structure: {structure}")
        
        improved_structure = anthropic_llm.improve_structure(structure=structure, _caller='create_book', response_format=BookStructure)
        logger.debug(f"Improved structure: {improved_structure}")
        
        if not isinstance(improved_structure, dict) or 'chapters' not in improved_structure:
            logger.error(f"Invalid improved structure: {improved_structure}")
            raise ValueError("Invalid book structure returned")
        
        chapters = {}
        for i, chapter in enumerate(improved_structure.get("chapters", [])[:10], 1):  # Limit to 10 chapters
            logger.debug(f"Processing chapter {i}: {chapter}")
            content = openai_llm.write_chapter(
                chapter=chapter["title"],
                topic=topic,
                points=chapter.get("main_points", []),
                _caller='create_book',
                response_format=ChapterContent
            )
            review = anthropic_llm.review_chapter(
                chapter=content,
                _caller='create_book',
                response_format=ChapterReview
            )
            rewritten_content = openai_llm.rewrite_chapter(
                original_chapter=content,
                review=review["review"],
                improvements=review["improvements"],
                _caller='create_book',
                response_format=ChapterContent
            )
            chapters[chapter["title"]] = {"content": rewritten_content, "review": review}
            
            # Save chapter to file
            os.makedirs("book", exist_ok=True)
            with open(f"book/chapter_{i:02d}.md", "w") as f:
                f.write(f"# {chapter['title']}\n\n{rewritten_content}")
        
        summary = openai_llm.summarize_book(chapters=chapters, _caller='create_book')
        
        logger.debug("Book creation completed")
        return {
            "structure": improved_structure,
            "chapters": chapters,
            "summary": summary
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
