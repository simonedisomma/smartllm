import functools
from typing import Callable, Optional, Dict, List
from .drivers.base import LLMDriver
from .driver_factory import DriverFactory
import networkx as nx
import matplotlib.pyplot as plt

class SmartLLM:
    def __init__(self, driver: Optional[str] = None, **driver_kwargs):
        if driver is None:
            self.driver = DriverFactory.create("openai", "gpt-4")
        else:
            self.driver = DriverFactory.create(driver, **driver_kwargs)
        self.functions: Dict[str, Callable] = {}
        self.function_calls: Dict[str, List[str]] = {}

    def configure(self, prompt: str, **kwargs):
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                formatted_prompt = prompt.format(**kwargs)
                result = self.driver.generate(formatted_prompt, **kwargs)
                
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

    def generate_flowchart(self, output_file: str = 'function_flowchart.png'):
        G = nx.DiGraph()
        
        for caller, called_functions in self.function_calls.items():
            G.add_node(caller)
            for func in called_functions:
                G.add_node(func)
                G.add_edge(caller, func)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=3000, font_size=10, font_weight='bold')
        
        edge_labels = {(u, v): '' for (u, v) in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.title("Function Call Flowchart")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def clear_function_calls(self):
        self.function_calls.clear()
