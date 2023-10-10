import ast
from collections import OrderedDict

IGNORE_FUNCTIONS = [
    "__init__",
    "setUp",
    "tearDown",
    "__len__",
    "__getitem__",
    "__iter__",
    "__enter__",
    "__exit__",
    "run_tests",
]


class PythonVisitor(ast.NodeVisitor):
    def __init__(self, filename, source):
        super().__init__()
        self.filename = filename
        self.source_code = source
        self.functions = OrderedDict()

    def visit_FunctionDef(self, node):
        # check blocklist to not index specified functions
        if node.name in IGNORE_FUNCTIONS:
            return

        # Python Dunder functions are never unittests
        if node.name.startswith("__") and node.name.endswith("__"):
            return

        # Check if there are any decorators.
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                # We do not want to index functions that are marked with
                # @property
                if decorator.id == "overload":
                    return

        signature = self.filename + ":" + node.name
        body = ast.get_source_segment(self.source_code, node)
        self.functions[signature] = body

    def visit_ClassDef(self, node):
        for sub_node in node.body:
            if isinstance(sub_node, ast.FunctionDef):
                # check blocklist to not index specified functions
                if sub_node.name in IGNORE_FUNCTIONS:
                    return

                # Python Dunder functions are never unittests
                if sub_node.name.startswith("__") and sub_node.name.endswith(
                    "__"
                ):
                    return

                # Check if there are any decorators.
                for decorator in sub_node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        # We do not want to index functions that are marked with
                        # @property
                        if decorator.id == "overload":
                            return

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


# funcs = get_functions("/home/osalpekar/pytorch/torch/distributed/utils.py")
# for func in funcs.keys():
#     print(func)
