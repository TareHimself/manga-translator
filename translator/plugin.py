from typing import Union

class PluginArgument:
    def __init__(self,name="",description="",required=True) -> None:
        self.name = name
        self.required = required
        self.description = description

    def get(self) -> dict[str,str]:
        return {
            "name": self.name,
            "description": self.description,
            "required": self.required
        }

class BasePlugin:
    def __init__(self) -> None:
        pass

    def get_name() -> str:
        return "unknown"
    
    def get_arguments() -> list[PluginArgument]:
        return []