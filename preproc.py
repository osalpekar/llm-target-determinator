import ast
from collections import OrderedDict

def get_functions(filename):
    with open(filename) as f:
        # TODO: filelist can just have relative paths and we can call
        # expanduser here just before opening the file
        file_content = f.read()

    functions = OrderedDict()

    module = ast.parse(file_content)

    # It takes about 36 minutes to do the ast function parsing for the
    # entire pytorch directory. This is likely prohibitive, so best if we
    # do it in dataloader.
    # By the end of this for-loop, we have a map of all functions in the
    # file and the function bodies
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef):
            # If the node is a function, extract its name and its arguments
            signature = filename + ":" + node.name
            body = ast.get_source_segment(file_content, node)
            functions[signature] = body
        elif isinstance(node, ast.ClassDef):
            # If the node is a class, we also want to get its methods
            for sub_node in node.body:
                if isinstance(sub_node, ast.FunctionDef):
                    signature = (
                        filename + ":" + node.name + "." + sub_node.name
                    )
                    body = ast.get_source_segment(file_content, sub_node)
                    functions[signature] = body

    return functions

