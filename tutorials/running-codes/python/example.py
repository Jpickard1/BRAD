"""
This is the top-level docstring for the example script.
It provides an overview of what the script does and any important details.

Example usage:
    python example.py
"""

def example_function():
    """
    This is the docstring for example_function.
    It explains what the function does and its parameters.
    """
    print("Hello, world!")

class ExampleClass:
    """
    This is the docstring for ExampleClass.
    It describes the purpose of the class and its methods.
    """
    def __init__(self):
        """
        This is the docstring for the constructor of ExampleClass.
        It describes the initialization process.
        """
        self.value = 42

    def example_method(self):
        """
        This is the docstring for example_method.
        It explains what this method does.
        """
        return self.value

if __name__ == "__main__":
    example_function()
    instance = ExampleClass()
    print(instance.example_method())
