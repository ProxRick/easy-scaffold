"""
Code execution tools for LLM function calling.

These tools execute code in sandboxed environments for safety.
"""

import logging
from typing import Dict, Any, Optional

from .manager import tool
from .sandbox import create_sandbox, SandboxResult

logger = logging.getLogger(__name__)


@tool(
    name="execute_python",
    description="Execute Python code in a sandboxed environment. Supports basic Python operations and safe mathematical computations.",
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute (e.g., '2 + 2', 'math.sqrt(16)', 'list(range(5))')"
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds (default: 5)",
                "default": 5
            },
            "sandbox_type": {
                "type": "string",
                "description": "Sandbox type: 'docker', 'modal', 'restricted', or 'subprocess' (default: 'subprocess')",
                "enum": ["docker", "modal", "restricted", "subprocess"],
                "default": "subprocess"
            }
        },
        "required": ["code"]
    }
)
async def execute_python(
    code: str,
    timeout: int = 5,
    sandbox_type: str = "subprocess"
) -> Dict[str, Any]:
    """
    Execute Python code in a sandboxed environment.
    
    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds
        sandbox_type: Type of sandbox to use
    
    Returns:
        Dictionary with execution result
    """
    try:
        sandbox = create_sandbox(sandbox_type)
        
        try:
            result: SandboxResult = await sandbox.execute(
                code=code,
                timeout=timeout,
                language="python"
            )
            
            if result.success:
                return {
                    "success": True,
                    "output": result.output,
                    "execution_time": result.execution_time
                }
            else:
                return {
                    "success": False,
                    "error": result.error,
                    "output": result.output if result.output else None
                }
        finally:
            # Cleanup if needed
            await sandbox.cleanup()
            
    except Exception as e:
        logger.error(f"Failed to execute Python code: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Sandbox initialization failed: {str(e)}"
        }


@tool(
    name="evaluate_expression",
    description="Safely evaluate a mathematical expression. Uses sandboxed execution for security.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', '2**3')"
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds (default: 2)",
                "default": 2
            }
        },
        "required": ["expression"]
    }
)
async def evaluate_expression(
    expression: str,
    timeout: int = 2
) -> Dict[str, Any]:
    """
    Evaluate a mathematical expression safely.
    
    Args:
        expression: Mathematical expression
        timeout: Execution timeout
    
    Returns:
        Dictionary with result
    """
    # Use restricted sandbox for math expressions (lighter weight)
    try:
        sandbox = create_sandbox("restricted", allowed_modules=["math"])
        
        try:
            result = await sandbox.execute(
                code=f"result = {expression}",
                timeout=timeout,
                language="python"
            )
            
            if result.success:
                # Extract result from output
                output = result.output
                if output.startswith("result = "):
                    output = output.replace("result = ", "").strip()
                
                return {
                    "success": True,
                    "result": output,
                    "expression": expression
                }
            else:
                return {
                    "success": False,
                    "error": result.error,
                    "expression": expression
                }
        finally:
            await sandbox.cleanup()
            
    except Exception as e:
        logger.error(f"Failed to evaluate expression: {e}")
        return {
            "success": False,
            "error": str(e),
            "expression": expression
        }


