import ast

def parse_python_functions(filename):
    with open(filename, "r") as source:
        tree = ast.parse(source.read())
        
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            print(f"Function {node.name} found at Line {node.lineno}:")
            
            # Extract the source lines of the function
            lines = []
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.Expr) and isinstance(child.value, ast.Str):  # Detect docstring
                    continue
                if isinstance(child, ast.arguments):
                    continue
                print(ast.dump(child))
                start_line = child.lineno
                end_line = child.end_lineno if hasattr(child, 'end_lineno') else start_line
                lines.extend(list(range(start_line, end_line + 1)))

            # Print the source lines of the function
            with open(filename, 'r') as f:
                all_lines = f.readlines()
                for line_num in lines:
                    print(all_lines[line_num - 1], end='')
            print("\n")

# Provide the filename here
parse_python_functions("/Users/sahanp/pytorch/torch/nn/modules/loss.py")