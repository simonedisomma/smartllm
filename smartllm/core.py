import functools
import inspect
import logging
from pydantic import BaseModel
from typing import Callable, Optional, Dict, List, Union, Type
from .drivers.base import LLMDriver
from .driver_factory import DriverFactory
from .visualization import graph

logger = logging.getLogger(__name__)

class SmartLLM:
    def __init__(self, provider_id: str, model_id: str):
        logger.debug(f"Initializing SmartLLM with provider_id: {provider_id}, model_id: {model_id}")
        self.driver = DriverFactory.create(provider_id, model_id)
        self.functions: Dict[str, Callable] = {}
        self.function_calls: Dict[str, List[str]] = {}
        logger.debug("SmartLLM initialized successfully")
        logger.debug(f"SmartLLM instance created with {provider_id} provider and {model_id} model")

    def configure(self, prompt: str, **kwargs):
        logger.debug(f"Configuring function with prompt: {prompt}")
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Get the caller's name automatically
                caller = inspect.currentframe().f_back.f_code.co_name
                response_format = kwargs.pop('response_format', None)
                logger.debug(f"Caller: {caller}, Response format: {response_format}")
                
                # Format the prompt
                formatted_prompt = prompt.format(**kwargs)
                logger.debug(f"Formatted prompt: {formatted_prompt}")
                
                # Generate the response
                logger.debug("Generating response from driver")
                result = self.driver.generate(formatted_prompt, response_format=response_format, **kwargs)
                logger.debug(f"Generated result: {result}")
                
                # Record the function call
                if caller not in self.function_calls:
                    self.function_calls[caller] = []
                self.function_calls[caller].append(func.__name__)
                logger.debug(f"Recorded function call: {caller} -> {func.__name__}")
                logger.debug(f"Function {func.__name__} called by {caller}")
                
                # If result is a string but we expected a Pydantic model, try to create an empty instance
                if isinstance(result, str) and response_format and issubclass(response_format, BaseModel):
                    logger.debug("Attempting to create empty Pydantic model instance")
                    try:
                        result = response_format()
                        logger.debug("Successfully created empty Pydantic model instance")
                    except:
                        logger.warning("Failed to create empty Pydantic model instance, using string result")
                
                # Call the original function with the LLM result
                logger.debug(f"Calling original function: {func.__name__}")
                return func(result, *args, **kwargs)
            
            self.functions[func.__name__] = wrapper
            logger.debug(f"Added function to SmartLLM: {func.__name__}")
            logger.debug(f"Function {func.__name__} configured with SmartLLM")
            return wrapper
        return decorator

    def __getattr__(self, name: str) -> Callable:
        logger.debug(f"Attempting to access attribute: {name}")
        if name in self.functions:
            logger.debug(f"Found function: {name}")
            logger.debug(f"Accessing configured function: {name}")
            return self.functions[name]
        logger.error(f"Attribute not found: {name}")
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def generate(self, prompt: str, response_format: Optional[Union[Type[BaseModel], str]] = None, **kwargs) -> str:
        logger.debug(f"Generating response for prompt: {prompt}")
        if not prompt.strip():
            logger.error("Empty prompt provided")
            raise ValueError("Prompt cannot be empty")
        # Remove '_caller' from kwargs before passing to driver.generate
        generate_kwargs = {k: v for k, v in kwargs.items() if k != '_caller'}
        logger.debug(f"Calling driver.generate with kwargs: {generate_kwargs}")
        logger.debug(f"Generating response for prompt: {prompt[:50]}...")  # Log first 50 chars of prompt
        return self.driver.generate(prompt, response_format=response_format, **generate_kwargs)

    def generate_flowchart(self, output_file: str = 'function_flowchart.png'):
        logger.debug(f"Generating flowchart, output file: {output_file}")
        graph.generate_flowchart(self.function_calls, output_file)
        logger.debug("Flowchart generation completed")
        logger.debug(f"Flowchart generated and saved to {output_file}")

    def clear_function_calls(self):
        logger.debug("Clearing function calls")
        self.function_calls.clear()
        logger.debug("Function calls cleared")
        logger.debug("Function call history has been cleared")

    def validate_params(self, func, **kwargs):
        """Validate parameters for a function call."""
        sig = inspect.signature(func)
        valid_params = {}
        for name, param in sig.parameters.items():
            if name in kwargs:
                valid_params[name] = kwargs[name]
            elif param.default is not inspect.Parameter.empty:
                valid_params[name] = param.default
            elif name not in ['llm_response', '_caller', 'response_format']:
                raise ValueError(f"Missing required parameter: {name}")
        return valid_params
