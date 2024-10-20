from typing import Dict, Type
from .drivers.base import LLMDriver
from .drivers import OpenAIDriver

class DriverFactory:
    _drivers: Dict[str, Type[LLMDriver]] = {
        "openai": OpenAIDriver
    }

    @classmethod
    def create(cls, driver_name: str, *args, **kwargs) -> LLMDriver:
        driver_class = cls._drivers.get(driver_name.lower())
        if driver_class is None:
            raise ValueError(f"Unknown driver: {driver_name}")
        return driver_class(*args, **kwargs)

    @classmethod
    def register_driver(cls, name: str, driver_class: Type[LLMDriver]):
        cls._drivers[name.lower()] = driver_class
