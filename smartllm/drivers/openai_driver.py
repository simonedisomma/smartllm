import openai
import json
from pydantic import BaseModel
from typing import Optional, Type, Union, Any, get_args, get_origin
from .base import LLMDriver

class OpenAIDriver(LLMDriver):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.client = openai.OpenAI()

    def generate(self, prompt: str, response_format: Optional[Union[Type[BaseModel], str]] = None, **kwargs) -> Any:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        try:
            # Remove any kwargs that are not supported by the OpenAI API
            valid_kwargs = {k: v for k, v in kwargs.items() if k in [
                'temperature', 'max_tokens', 'top_p', 'frequency_penalty',
                'presence_penalty', 'stop', 'n', 'stream', 'logit_bias'
            ]}

            # Append JSON format instruction to the prompt
            json_instruction = self._get_json_instruction(response_format)
            full_prompt = f"{prompt}\n\n{json_instruction}"

            messages = [
                {"role": "system", "content": "You are a helpful assistant. Please provide your response in JSON format."},
                {"role": "user", "content": full_prompt}
            ]

            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                return self._generate_structured(messages, response_format, **valid_kwargs)
            elif response_format == "json":
                return self._generate_json(messages, **valid_kwargs)
            else:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    **valid_kwargs
                )
                return response.choices[0].message.content.strip()
        except openai.BadRequestError as e:
            error_message = f"Bad Request Error: {str(e)}"
            print(error_message)  # Print for debugging
            return error_message
        except Exception as e:
            error_message = f"Error: {str(e)}"
            print(error_message)  # Print for debugging
            return error_message

    def _get_json_instruction(self, response_format: Optional[Union[Type[BaseModel], str]]) -> str:
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            fields = response_format.model_fields
            if len(fields) == 1:
                key, field = next(iter(fields.items()))
                return f"Return the result as a JSON object with a key '{key}' containing a {self._get_field_type_description(field)}."
            else:
                field_descriptions = [f"'{k}': {self._get_field_type_description(v)}" for k, v in fields.items()]
                return f"Return the result as a JSON object with the following structure: {{{', '.join(field_descriptions)}}}."
        elif response_format == "json":
            return "Return the result as a JSON object."
        else:
            return "Return the result as a JSON object if possible."

    def _get_field_type_description(self, field):
            field_type = field.annotation
            if get_origin(field_type) is list:
                item_type = get_args(field_type)[0]
                return f"list of {item_type.__name__.lower()}s"
            else:
                return field_type.__name__.lower()

    def _generate_structured(self, messages, response_format: Type[BaseModel], **kwargs):
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                response_format={"type": "json_object"},
                **kwargs
            )
            content = response.choices[0].message.content.strip()
            parsed_content = json.loads(content)
            
            # Check if the parsed content matches the expected structure
            if set(parsed_content.keys()) != set(response_format.model_fields.keys()):
                # If not, try to adapt the content to match the expected structure
                adapted_content = self._adapt_content(parsed_content, response_format)
                return response_format(**adapted_content)
            
            return response_format(**parsed_content)
        except json.JSONDecodeError as e:
            error_message = f"Error: Unable to parse JSON response. {str(e)}"
            print(error_message)  # Print for debugging
            return error_message
        except Exception as e:
            error_message = f"Error in structured generation: {str(e)}"
            print(error_message)  # Print for debugging
            return error_message

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

    def _generate_json(self, messages, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                response_format={"type": "json_object"},
                **kwargs
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except json.JSONDecodeError as e:
            error_message = f"Error: Unable to parse JSON response. {str(e)}"
            print(error_message)  # Print for debugging
            return error_message
        except Exception as e:
            error_message = f"Error in JSON generation: {str(e)}"
            print(error_message)  # Print for debugging
            return error_message
