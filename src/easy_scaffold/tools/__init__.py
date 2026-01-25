"""
Tools module for LLM function calling.

Public API:
- ToolRegistry: Tool discovery and registration
- ToolExecutor: Tool execution engine
- tool: Decorator for defining tools
- Sandbox: Sandbox implementations for safe code execution
"""

from .manager import ToolRegistry, ToolExecutor, tool, get_registry
from .sandbox import create_sandbox, Sandbox, SandboxResult

# Import tool modules to register tools
try:
    from . import math  # noqa: F401
except ImportError:
    pass  # math tools optional

try:
    from . import code  # noqa: F401
except ImportError:
    pass  # code tools optional

__all__ = [
    "ToolRegistry",
    "ToolExecutor",
    "tool",
    "get_registry",
    "create_sandbox",
    "Sandbox",
    "SandboxResult",
]

