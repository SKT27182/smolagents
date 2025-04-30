# This file is currently empty as we're using smolagents' built-in tools
# We'll implement custom tools here later when needed
from smolagents import Tool


class CalculatorTool(Tool):

    name = "calculator"
    description = """This is a calculator tool. Use it to perform basic arithmetic operations. 
        You can ask it to add, subtract, multiply, or divide numbers. It returns the numeric output of the operation."""

    inputs = {
        "num1": {
            "type": "number",
            "description": "The first number to be used in the operation.",
        },
        "num2": {
            "type": "number",
            "description": "The second number to be used in the operation.",
        },
        "operation": {
            "type": "string",
            "description": "The operation to perform. Can be 'add', 'sub', 'mul', or 'div'.",
        },
    }
    output_type = "number"

    def forward(self, num1: float, num2: float, operation: str) -> float:
        import operator

        operator_dict = {
            "add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "div": operator.truediv,
        }

        if operation not in operator_dict:
            raise ValueError(
                f"Invalid operation: {operation}. Must be one of {list(operator_dict.keys())}."
            )

        return operator_dict[operation](num1, num2)
