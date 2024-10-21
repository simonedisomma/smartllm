import functools
from typing import Callable, Optional, Dict, List, Union, Type
from .drivers.base import LLMDriver
from .driver_factory import DriverFactory
from pydantic import BaseModel
from .visualization import graph

class SmartLLM:
    def __init__(self, provider_id: str, model_id: str):
        self.driver = DriverFactory.create(provider_id, model_id)
        self.functions: Dict[str, Callable] = {}
        self.function_calls: Dict[str, List[str]] = {}

    def configure(self, prompt: str, **kwargs):
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Remove 'name' and '_caller' from kwargs before passing to generate
                formatted_prompt = prompt.format(**kwargs)
                generate_kwargs = {k: v for k, v in kwargs.items() if k not in ['name', '_caller']}
                
                # Extract response_format if provided
                response_format = generate_kwargs.pop('response_format', None)
                
                result = self.driver.generate(formatted_prompt, response_format=response_format, **generate_kwargs)
                
                # Record the function call
                caller = kwargs.get('_caller', 'main')
                if caller not in self.function_calls:
                    self.function_calls[caller] = []
                self.function_calls[caller].append(func.__name__)
                
                # Call the original function with the LLM result
                return func(result, *args, **kwargs)
            
            self.functions[func.__name__] = wrapper
            return wrapper
        return decorator

    def __getattr__(self, name: str) -> Callable:
        if name in self.functions:
            return self.functions[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def generate(self, prompt: str, response_format: Optional[Union[Type[BaseModel], str]] = None, **kwargs) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        # Remove '_caller' from kwargs before passing to driver.generate
        generate_kwargs = {k: v for k, v in kwargs.items() if k != '_caller'}
        return self.driver.generate(prompt, response_format=response_format, **generate_kwargs)

    def generate_flowchart(self, output_file: str = 'function_flowchart.png'):
        graph.generate_flowchart(self.function_calls, output_file)

    def clear_function_calls(self):
        self.function_calls.clear()
