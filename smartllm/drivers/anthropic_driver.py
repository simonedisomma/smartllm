import json
import logging
from typing import Union, Type
from pydantic import BaseModel
from anthropic import Anthropic
from .base import LLMDriver

logger = logging.getLogger(__name__)

class AnthropicDriver(LLMDriver):
    def __init__(self, model: str = "claude-3-sonnet-20240229"):
        self.model = model
        self.client = Anthropic()
        logger.debug(f"AnthropicDriver initialized with model: {self.model}")

    def generate(self, prompt: str, response_format: Union[Type[BaseModel], str, None] = None, **kwargs) -> Union[str, BaseModel]:
        logger.info(f"Anthropic LLM Call: model={self.model}")
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"Additional kwargs: {kwargs}")

        if response_format and issubclass(response_format, BaseModel):
            logger.debug("Using Pydantic model for response format")
            json_structure = response_format.schema_json()
            formatted_prompt = f"{prompt}\n\nPlease provide your response as a valid JSON object that matches this structure:\n{json_structure}\n\nDo not include the schema in your response, only the data."
            
            logger.debug(f"Formatted prompt: {formatted_prompt}")
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": formatted_prompt},
                    {"role": "assistant", "content": "Here is the JSON response:\n{"}
                ]
            ).content[0].text
            
            logger.debug(f"Raw response from Anthropic: {message}")

            try:
                # Ensure the response starts with '{' and ends with '}'
                if not message.strip().startswith('{'):
                    message = '{' + message
                if not message.strip().endswith('}'):
                    message = message + '}'
                
                # Parse the JSON response
                response_dict = json.loads(message, strict=False)
                
                logger.debug(f"Parsed JSON response: {response_dict}")
                return response_format(**response_dict)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                logger.warning(f"Raw response causing the error: {message}")
                return response_format()
            except Exception as e:
                logger.error(f"Error processing response: {e}")
                return response_format()
        else:
            logger.debug("No specific response format requested")
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            ).content[0].text
            logger.debug(f"Generated response: {message}")
            return message
