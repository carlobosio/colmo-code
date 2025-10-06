import numpy as np
import re
FLOAT = re.compile(r"(?<![^ ])[-+]?\d*\.\d+|(?<![^ ])[-+]?\d+")
# Updated regex to handle scientific notation and better handle word boundaries
# FLOAT = re.compile(r'\b[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\b')

class ProgramWrapper:
    def __init__(self, program: str, significant_digits: int = 3):  # Added significant_digits parameter
        """Initialize ProgramWrapper with a code snippet.
        
        Args:
            program: String containing the code snippet to process
            significant_digits: Number of significant digits for float values
        """
        self.program = program
        self.significant_digits = significant_digits
        self.floats = self.extract_floats()
        self.num_floats = len(self.floats)

    def extract_floats(self) -> list[float]:
        """Extracts all floats from a string.
        
        Returns:
            List of float values found in the program
        """
        return [float(x) for x in re.findall(FLOAT, self.program)]
    
    def format_number(self, x: float) -> str:
        """Format a float to the specified number of significant digits."""
        formatted = f"{x:.{self.significant_digits}f}".rstrip('0').rstrip('.')
        return formatted
    
    def sub_floats(self, numbers: list[float] | np.ndarray) -> str:
        """Substitutes all floats in the program with new values.
        
        Args:
            numbers: List of new float values to substitute
            
        Returns:
            Modified program string
            
        Raises:
            ValueError: If length of numbers doesn't match number of floats in program
        """
        if isinstance(numbers, np.ndarray):
            numbers = numbers.tolist()

        if len(numbers) != self.num_floats:
            raise ValueError(
                f"Number of replacements ({len(numbers)}) does not match "
                f"number of floats in program ({self.num_floats})"
            )
        
        # Convert all numbers to float and format them
        # formatted_numbers = [float(self.format_number(x)) for x in numbers]
        
        # Update internal float list with formatted numbers
        self.floats = numbers
        
        # Create string replacements
        replacement = [self.format_number(x) for x in numbers]
        replacement_iter = iter(replacement)

        def replace_with_next(match: re.Match) -> str:
            return next(replacement_iter)

        self.program = re.sub(FLOAT, replace_with_next, self.program)
        self.floats = numbers
        return self.program

    def sub_params(self) -> str:
        # Create string replacements
        replacement = [f'params[{i}]' for i in range(len(self.floats))]
        replacement_iter = iter(replacement)
        def replace_with_next(match: re.Match) -> str:
            return next(replacement_iter)

        self.program = re.sub(FLOAT, replace_with_next, self.program)
        return self.program

    def get_program(self) -> str:
        return self.program
    
    def get_floats(self) -> list[float]:
        return self.floats
    
    def get_num_floats(self) -> int:
        return self.num_floats

if __name__ == "__main__":
    import numpy as np
    # from zoopt import 

    # prompt = ("def heuristic(obs: np.ndarray) -> float:\n\n"
    #         "\"\"\"Returns an action between -1 and 1.\n"
    #         "obs size is 3.\n"
    #         "\"\"\"\n"
    #         "\tx1 = np.arctan2(-obs[1], obs[0])\n"
    #         "\tx2 = obs[2]\n"
    #         "\tif x1 < 0 and x2 > 0:\n"
    #         "\t\taction += 1\n"
    #         "\telif x1 > 0 and x2 < 0:\n"
    #         "\t\taction -= 1\n"
    #         "\telif abs(x1) >= abs(x2):\n"
    #         "\t\taction = np.sign(x1)\n"
    #         "\telse:\n"
    #         "\t\taction = np.sign(x2)\n\n"
    #         "\treturn action")
    function = ("def function(x1, x2) -> float:\n"
                "\treturn 2.5*x1**2 + 3.5*x2**2\n\n"
                "result = function(x1, x2)")
    
    p = ProgramWrapper(function)
    print(p.get_program())
    loc = {'x1': 2, 'x2': np.sqrt(2)}
    exec(p.get_program(), {'np': np}, loc)
    print(loc['result'])
    print(p.get_floats())
    replacement = [1.0,1.0]
    p.sub_floats(replacement)
    print(p.get_program())
    exec(p.get_program(), {'np': np}, loc)
    print(loc['result'])
    print(p.get_floats())
    