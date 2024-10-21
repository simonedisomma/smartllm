import json
import logging
from typing import Union, Type, Any
from pydantic import BaseModel
from anthropic import Anthropic
from .base import LLMDriver

logger = logging.getLogger(__name__)

class AnthropicDriver(LLMDriver):
    def __init__(self, model: str = "claude-3-sonnet-20240229"):
        self.model = model
        self.client = Anthropic()
        logger.debug(f"AnthropicDriver initialized with model: {self.model}")

    def generate(self, prompt: str, response_format: Union[Type[BaseModel], str, None] = None, **kwargs) -> Union[str, dict]:
        logger.info(f"Anthropic LLM Call: model={self.model}")
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"Additional kwargs: {kwargs}")

        if response_format and issubclass(response_format, BaseModel):
            logger.debug("Using Pydantic model for response format")
            json_structure = response_format.model_json_schema()
            formatted_prompt = f"{prompt}\n\nPlease provide your response as a valid JSON object that matches this structure:\n{json.dumps(json_structure, indent=2)}\n\nDo not include the schema in your response, only the data."
            
            logger.debug(f"Formatted prompt: {formatted_prompt}")
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": formatted_prompt},
                    {"role": "assistant", "content": "Here is the JSON response:"}
                ]
            ).content[0].text
            
            logger.debug(f"Raw response from Anthropic: {message}")

            try:
                # Extract JSON from the message
                json_start = message.find('{')
                json_end = message.rfind('}') + 1
                if json_start == -1 or json_end == 0:
                    raise ValueError("No valid JSON found in the response")
                json_str = message[json_start:json_end]
                
                # Parse the JSON response, allowing newlines in strings
                response_dict = json.loads(json_str, strict=False)
                
                logger.debug(f"Parsed JSON response: {response_dict}")
                # Return the parsed JSON dictionary
                return response_dict
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response causing the error: {message}")
                # Instead of raising an error, return the raw message
                return {"content": message}
            except Exception as e:
                logger.error(f"Error processing response: {e}")
                # Instead of raising an error, return the raw message
                return {"content": message}
        else:
            logger.debug("No specific response format requested")
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            ).content[0].text
            logger.debug(f"Generated response: {message}")
            return message

    def _adapt_content(self, content: dict, response_format: Type[BaseModel]) -> dict:
        adapted_content = {}
        for field_name in response_format.model_fields:
            if field_name in content:
                adapted_content[field_name] = content[field_name]
            else:
                # Try to find a matching key
                for key in content.keys():
                    if field_name.lower() in key.lower():
                        adapted_content[field_name] = content[key]
                        break
        return adapted_content
