import ast


class CodeExecutionEngine:
    def __init__(self):
        self.blacklist = [
            # Disallowed built-in functions
            'compile', 'exec', 'eval', 'input',

            # Disallowed built-in objects
            'globals', 'locals', 'vars',

            # Disallowed modules
            'os', 'subprocess', 'shutil', 'tempfile', 'socket',

            # Disallowed object attributes
            '__import__', '__subclasses__', '__bases__', '__class__', '__mro__'
        ]

    def is_malicious(self, input_code):
        # Parse the input code as an AST to check for any forbidden operations

        ast.parse(input_code, mode='exec')
        for node in ast.walk(ast.parse(input_code)):
            if isinstance(node, ast.Name) and node.id in self.blacklist:
                return True
        return False

    def balance_quotes_and_brackets(self, input_string):
        stack = []
        for i, c in enumerate(input_string):
            if c in ['(', '{', '[', "'", '"']:
                stack.append((c, i))
            elif c in [')', '}', ']', "'", '"']:
                if not stack:
                    # unbalanced quotes or brackets
                    return None
                top, pos = stack.pop()
                if top == '(' and c != ')':
                    return None
                elif top == '{' and c != '}':
                    return None
                elif top == '[' and c != ']':
                    return None
                elif top in ["'", '"'] and top != c:
                    return None
        if stack:
            # unbalanced quotes or brackets
            return None
        # if everything is balanced, return the original string
        return input_string

    def forward(self, code):
        code = self.balance_quotes_and_brackets(code)
        alpha = None
        exec(code)
        return alpha




