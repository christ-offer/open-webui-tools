"""
title: Calculator Tool
description: A simple calculator tool that supports basic arithmetic operations, exponentiation, and mathematical functions.
author: Pyotr Growpotkin
author_url: https://github.com/christ-offer/
github: https://github.com/christ-offer/open-webui-tools
funding_url: https://github.com/open-webui
version: 0.0.2
license: MIT
"""

import math
import re
from typing import Union, Dict, Callable


class Tools:
    def __init__(self):
        self.operations: Dict[str, Callable] = {
            "+": lambda x, y: x + y,
            "-": lambda x, y: x - y,
            "*": lambda x, y: x * y,
            "/": lambda x, y: x / y if y != 0 else float("inf"),
            "^": lambda x, y: x**y,
            "sqrt": lambda x: math.sqrt(x) if x >= 0 else float("nan"),
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": lambda x: math.log(x) if x > 0 else float("nan"),
        }

    def calculator(self, equation: str) -> str:
        """
        Calculate the result of an equation using a safe evaluation method.

        This calculator supports basic arithmetic operations (+, -, *, /),
        exponentiation (^), and some mathematical functions (sqrt, sin, cos, tan, log).

        :param equation: The equation to calculate.
        :return: A string representing the equation and its result.

        :raises ValueError: If the equation is invalid or contains unsupported operations.

        Examples:
        >>> tools = Tools()
        >>> tools.calculator("2 + 2")
        '2 + 2 = 4.0'
        >>> tools.calculator("sin(0.5)")
        'sin(0.5) = 0.479425538604203'
        >>> tools.calculator("2 ^ 3")
        '2 ^ 3 = 8.0'
        """
        try:
            # Remove all whitespace from the equation
            equation = re.sub(r"\s+", "", equation)

            # Tokenize the equation
            tokens = re.findall(
                r"(\d+\.?\d*|[+\-*/^()]|sqrt|sin|cos|tan|log)", equation
            )

            # Convert infix notation to Reverse Polish Notation (RPN)
            rpn = self._shunting_yard(tokens)

            # Evaluate RPN
            result = self._evaluate_rpn(rpn)

            return f"{equation} = {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _shunting_yard(self, tokens: list) -> list:
        """
        Convert infix notation to Reverse Polish Notation (RPN) using the shunting yard algorithm.
        """
        output = []
        operators = []
        precedence = {
            "+": 1,
            "-": 1,
            "*": 2,
            "/": 2,
            "^": 3,
            "sqrt": 4,
            "sin": 4,
            "cos": 4,
            "tan": 4,
            "log": 4,
        }

        for token in tokens:
            if token.replace(".", "").isdigit():
                output.append(token)
            elif token in self.operations:
                while (
                    operators
                    and operators[-1] != "("
                    and precedence.get(operators[-1], 0) >= precedence.get(token, 0)
                ):
                    output.append(operators.pop())
                operators.append(token)
            elif token == "(":
                operators.append(token)
            elif token == ")":
                while operators and operators[-1] != "(":
                    output.append(operators.pop())
                if operators and operators[-1] == "(":
                    operators.pop()
                else:
                    raise ValueError("Mismatched parentheses")

        while operators:
            if operators[-1] == "(":
                raise ValueError("Mismatched parentheses")
            output.append(operators.pop())

        return output

    def _evaluate_rpn(self, rpn: list) -> float:
        """
        Evaluate a mathematical expression in Reverse Polish Notation (RPN).
        """
        stack = []
        for token in rpn:
            if token.replace(".", "").isdigit():
                stack.append(float(token))
            elif token in self.operations:
                if token in ("sqrt", "sin", "cos", "tan", "log"):
                    if len(stack) < 1:
                        raise ValueError(f"Not enough operands for {token}")
                    x = stack.pop()
                    stack.append(self.operations[token](x))
                else:
                    if len(stack) < 2:
                        raise ValueError(f"Not enough operands for {token}")
                    y, x = stack.pop(), stack.pop()
                    stack.append(self.operations[token](x, y))
            else:
                raise ValueError(f"Unknown token: {token}")

        if len(stack) != 1:
            raise ValueError("Invalid expression")

        return stack[0]
