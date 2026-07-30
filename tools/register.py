from typing import Callable, Dict, Any, Optional, List, get_type_hints
import inspect
import json

# 保留的开发备注。
TYPE_MAP = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null"
}

class FunctionRegistry:
    """
    A registry class for functions.
    Stores functions and their metadata, and can generate JSON Schema
    for use with large language models (LLMs).
    """
    def __init__(self):
        # 保存中间状态。
        """初始化对象状态。"""
        self.functions: Dict[str, Callable] = {}
        # 保存中间状态。
        self.metadata: Dict[str, dict] = {}

    def register(self,
                 name: Optional[str] = None,          # 保留的开发备注。
                 description: str = "",               # 保留的开发备注。
                 parameters: Optional[List[Dict]] = None,  # 参数说明。
                 constraints: str = "",               # 参数说明。
                 returns: Optional[Dict] = None):    # 返回值说明。
        """处理 register 相关逻辑。"""
        def decorator(func: Callable):
            """处理 decorator 相关逻辑。"""
            nonlocal name
            if name is None:
                name = func.__name__  # 默认值说明。

            # 构建中间结果。
            properties = {}  # 保存中间状态。
            required = []    # 参数说明。
            for p in parameters:
                param_name = p["name"]
                properties[param_name] = {
                    "type": p["type"],                # 参数说明。
                    "description": p.get("description"),  # 参数说明。
                }
                if p.get("required", True):  # 默认值说明。
                    required.append(param_name)

            schema = {
                "type": "object",     # 参数说明。
                "properties": properties,
                "required": required
            }

            # 返回值说明。
            if returns:
                return_info = returns  # 返回值说明。
            else:
                # 返回值说明。
                return_type = TYPE_MAP.get(get_type_hints(func).get('return'), 'object')
                return_info = {
                    "type": return_type
                }

            # 保留的开发备注。
            self.functions[name] = func
            # 保留的开发备注。
            self.metadata[name] = {
                "name": name,              # 保留的开发备注。
                "description": description,  # 保留的开发备注。
                "parameters": schema,        # 参数说明。
                "constraints": constraints,  # 保留的开发备注。
                "returns": return_info       # 返回值说明。
            }
            return func  # 返回值说明。
        return decorator

    def get_function(self, name: str) -> Callable:
        """获取 get function 相关信息。"""
        return self.functions.get(name)

    def get_metadata(self, name: str) -> dict:
        """获取 get metadata 相关信息。"""
        return self.metadata.get(name)

    def list_functions(self) -> list:
        """处理 list functions 相关逻辑。"""
        return list(self.functions.keys())

    def export_tool_schemas(self) -> list:
        """导出 export tool schemas 相关结果。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": meta["name"],
                    "description": meta["description"],
                    "constraints": meta["constraints"],
                    "parameters": meta["parameters"],
                    "returns": meta["returns"]
                }
            }
            for meta in self.metadata.values()
        ]

# 保留的开发备注。
function_register = FunctionRegistry()
