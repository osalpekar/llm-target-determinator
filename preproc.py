import ast
from collections import OrderedDict


class PythonVisitor(ast.NodeVisitor):
    def __init__(self, filename, source):
        super().__init__()
        self.filename = filename
        self.source_code = source
        self.functions = OrderedDict()

    def visit_FunctionDef(self, node):
        signature = self.filename + ":" + node.name
        body = ast.get_source_segment(self.source_code, node)
        self.functions[signature] = body

    def visit_ClassDef(self, node):
        for sub_node in node.body:
            if isinstance(sub_node, ast.FunctionDef):
                # TODO: add a blocklist to not index functions like __init__,
                # setUp, and tearDown
                signature = (
                    self.filename + ":" + node.name + "." + sub_node.name
                )
                body = ast.get_source_segment(self.source_code, sub_node)
                self.functions[signature] = body

def get_functions(filename):
    # It takes about 36 minutes to do the ast function parsing for the
    # entire pytorch directory. This is likely prohibitive, so best if we
    # do it in dataloader.
    # By the end of this function, we have a map of all functions in the
    # file and the function bodies
    with open(filename) as f:
        # TODO: filelist can just have relative paths and we can call
        # expanduser here just before opening the file
        file_content = f.read()

    functions = OrderedDict()

    visitor = PythonVisitor(filename, file_content)
    tree = ast.parse(file_content)
    visitor.visit(tree)

    return visitor.functions
