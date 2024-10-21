from typing import Dict, Type
from .drivers.base import LLMDriver
from .drivers import OpenAIDriver
from .drivers import AnthropicDriver

class DriverFactory:
    _drivers: Dict[str, Type[LLMDriver]] = {
        "openai": OpenAIDriver,
        "anthropic": AnthropicDriver
    }

    @classmethod
    def create(cls, provider_id: str, model_id: str, *args, **kwargs) -> LLMDriver:
        driver_class = cls._drivers.get(provider_id.lower())
        if driver_class is None:
            raise ValueError(f"Unknown driver: {provider_id}")
        return driver_class(model_id, *args, **kwargs)

    @classmethod
    def register_driver(cls, name: str, driver_class: Type[LLMDriver]):
        cls._drivers[name.lower()] = driver_class
