from typing import Union


class PluginArgumentType:
    TEXT = 0
    SELECT = 1

class PluginArgument:
    def __init__(self,name: str,description: str) -> None:
        self.name = name
        self.type = PluginArgumentType.TEXT
        self.description = description

    def get(self) -> dict[str,str]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
        }
    
class PluginTextArgument(PluginArgument):
    def __init__(self, name: str,description: str) -> None:
        super().__init__(name, description)
        self.type = PluginArgumentType.TEXT

class PluginSelectArgumentOption:
    def __init__(self,name: str,value: str) -> None:
        self.name = name
        self.value = value

    def get(self) -> dict[str, str]:
        return {
            "name": self.name,
            "value": self.value,
        }

class PluginSelectArgument(PluginArgument):
    def __init__(self, name: str,description: str,options: list[PluginSelectArgumentOption]) -> None:
        super().__init__(name, description)
        self.type = PluginArgumentType.SELECT
        self.options = options

    def get(self) -> dict[str, str]:
        data = super().get()
        data['options'] = [x.get() for x in self.options]
        return data

class BasePlugin:
    def __init__(self) -> None:
        pass

    def get_name() -> str:
        return "unknown"
    
    def get_arguments() -> list[PluginArgument]:
        return []
    
    def is_valid() -> bool:
        return True