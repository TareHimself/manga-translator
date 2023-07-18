from typing import Union


class PluginArgumentType:
    TEXT = 0
    SELECT = 1

class PluginArgument:
    def __init__(self,id: str,name: str,description: str,default: str="") -> None:
        self.id = id
        self.name = name
        self.type = PluginArgumentType.TEXT
        self.description = description
        self.default = default

    def get(self) -> dict[str,str]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "default": self.default,
            "type": self.type,
        }
    
class PluginTextArgument(PluginArgument):
    def __init__(self, id: str,name: str,description: str,default: str="") -> None:
        super().__init__(id,name,description,default)
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
    def __init__(self,id: str,name: str,description: str,options: list[PluginSelectArgumentOption] ,default: str="") -> None:
        super().__init__(id,name,description,default)
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