"""
Mathematical tools for LLM function calling.

Example tools for performing calculations and mathematical operations.
Uses sandboxed execution for safety.
"""

import logging
from typing import Dict, Any

from .manager import tool
from .sandbox import create_sandbox

logger = logging.getLogger(__name__)


@tool(
    name="calculate",
    description="Evaluate a mathematical expression safely. Supports basic arithmetic, powers, and common functions.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', '2^3')"
            }
        },
        "required": ["expression"]
    }
)
async def calculate(expression: str) -> Dict[str, Any]:
    """
    Evaluate a mathematical expression using sandboxed execution.
    
    Args:
        expression: Mathematical expression as a string
    
    Returns:
        Dictionary with result and expression
    """
    try:
        # Use restricted sandbox for math (lightweight, fast)
        sandbox = create_sandbox("restricted", allowed_modules=["math"])
        
        try:
            # Wrap expression in eval
            code = f"result = {expression}"
            result = await sandbox.execute(code, timeout=2, language="python")
            
            if result.success:
                # Extract numeric result from output
                output = result.output
                if "result = " in output:
                    output = output.split("result = ")[-1].strip()
                
                # Try to convert to number
                try:
                    numeric_result = float(output)
                    if numeric_result.is_integer():
                        numeric_result = int(numeric_result)
                except ValueError:
                    numeric_result = output
                
                return {
                    "result": numeric_result,
                    "expression": expression,
                    "success": True
                }
            else:
                return {
                    "result": None,
                    "expression": expression,
                    "success": False,
                    "error": result.error
                }
        finally:
            await sandbox.cleanup()
            
    except Exception as e:
        logger.warning(f"Failed to evaluate expression '{expression}': {e}")
        return {
            "result": None,
            "expression": expression,
            "success": False,
            "error": str(e)
        }


@tool(
    name="solve_equation",
    description="Solve a simple linear equation of the form ax + b = c",
    parameters={
        "type": "object",
        "properties": {
            "a": {
                "type": "number",
                "description": "Coefficient of x"
            },
            "b": {
                "type": "number",
                "description": "Constant term"
            },
            "c": {
                "type": "number",
                "description": "Right-hand side value"
            }
        },
        "required": ["a", "b", "c"]
    }
)
async def solve_equation(a: float, b: float, c: float) -> Dict[str, Any]:
    """
    Solve linear equation ax + b = c.
    
    Args:
        a: Coefficient of x
        b: Constant term
        c: Right-hand side
    
    Returns:
        Dictionary with solution
    """
    try:
        if a == 0:
            return {
                "success": False,
                "error": "Coefficient 'a' cannot be zero (not a linear equation)"
            }
        
        x = (c - b) / a
        
        return {
            "success": True,
            "solution": x,
            "equation": f"{a}x + {b} = {c}",
            "verification": a * x + b == c
        }
    except Exception as e:
        logger.warning(f"Failed to solve equation: {e}")
        return {
            "success": False,
            "error": str(e)
        }


