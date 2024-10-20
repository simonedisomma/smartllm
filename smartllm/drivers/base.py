from abc import ABC, abstractmethod

class LLMDriver(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass
