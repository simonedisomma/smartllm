import logging
from pydantic import BaseModel, Field
from smartllm import SmartLLM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = SmartLLM(provider_id="openai", model_id="chatgpt-4o-latest")

class BookStructure(BaseModel):
    title: str = Field(description="Book title")
    chapters: list[str] = Field(description="List of chapter titles")

class ChapterContent(BaseModel):
    content: str = Field(description="Content of the chapter")

@llm.configure("Create a simple structure for a book about {topic}. Include a title and a list of 3-5 chapter titles.")
def create_book_structure(llm_response: BookStructure, topic: str) -> dict:
    logger.info(f"Creating book structure for topic: {topic}")
    return llm_response.model_dump()

@llm.configure("Write a brief chapter for '{chapter}' in the book about {topic}.")
def write_chapter(llm_response: ChapterContent, chapter: str, topic: str) -> str:
    logger.info(f"Writing chapter: {chapter}")
    return llm_response.content

def create_book(topic: str) -> dict:
    try:
        structure = llm.create_book_structure(topic=topic, response_format=BookStructure)
        logger.info(f"Book structure: {structure}")

        chapters = {}
        for chapter in structure['chapters']:
            content = llm.write_chapter(chapter=chapter, topic=topic, response_format=ChapterContent)
            chapters[chapter] = content

        return {
            "structure": structure,
            "chapters": chapters
        }
    except Exception as e:
        logger.error(f"Error in create_book: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    book = create_book("Artificial Intelligence Ethics")
    if "error" not in book:
        print(f"Book structure: {book['structure']}")
        print(f"Number of chapters: {len(book['chapters'])}")
        for title, content in book['chapters'].items():
            print(f"\nChapter: {title}")
            print(f"Content preview: {content[:100]}...")
    else:
        print(f"Error: {book['error']}")
