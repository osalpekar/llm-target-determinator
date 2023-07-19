import ast
import inspect
import pprint
def get_functions_from_file(filename):
    with open(filename, 'r') as file:
        content = file.read()
    
    module = ast.parse(content)
    
    functions = {}
    
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef):
            # If the node is a function, extract its name and its arguments
            signature = node.name
            body = ast.get_source_segment(content, node)
            functions[signature] = body
        elif isinstance(node, ast.ClassDef):
            # If the node is a class, we also want to get its methods
            for sub_node in node.body:
                if isinstance(sub_node, ast.FunctionDef):
                    signature = node.name + '.' + sub_node.name
                    body = ast.get_source_segment(content, sub_node)
                    functions[signature] = body
                    
    return functions


# Provide the filename here
output = get_functions_from_file("/Users/sahanp/pytorch/torch/nn/modules/loss.py")
pprint.pprint(output)