"""
title: Python Code Interpreter
author: Pyotr Growpotkin
version: 0.0.1
"""

import subprocess
import json


class Tools:
    def __init__(self):
        pass

    def execute_python_code(self, code: str) -> str:
        """
        Execute the given Python code and return the output.

        :param code: The Python code to execute.
        :return: The output of the executed code.
        """
        try:
            # Execute the Python code
            result = subprocess.run(
                ["python", "-c", code], capture_output=True, text=True, check=True
            )
            stdout = result.stdout.strip()

            # Return the result as a JSON string
            return json.dumps({"result": stdout})
        except subprocess.CalledProcessError as e:
            return json.dumps({"error": e.output.strip()})
        except Exception as e:
            return json.dumps({"error": str(e)})
