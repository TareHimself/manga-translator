# This file is an attempt to extract a given function or name from a module with all its depencencies. it has trouble with relative imports and needs more work
import ast
import os
import re
import argparse
import pkg_resources
import collections
from typing import Any
from typing import Union, Generic, TypeVar
from functools import cmp_to_key
from collections import OrderedDict
from ast import AnnAssign, Attribute, Call, ClassDef, Expr, FunctionDef, Name

T = TypeVar('T')

class OrderedSet(Generic[T]):
    def __init__(self, iterable=None):
        self._data = []
        if iterable is not None:
            self._data = list(iterable)

    def __len__(self):
        return len(self._data)

    def __contains__(self, item: T):
        return item in self._data

    def add(self, item: T):
        if item not in self._data:
            self._data.append(item)

    def discard(self, item: T):
        if item in self:
            self._data.remove(item)

    def update(self,other: 'OrderedSet'):
        for x in other:
            self.add(x)

    def difference_update(self,other):
        for x in other:
            self.discard(x)

    def intersection(self,other):
        result = OrderedSet()
        for x in other:
            if x in self:
                result.add(x)
        
        return result

    def __iter__(self):
        return iter(self._data)

    def __reversed__(self):
        return reversed(self._data)

    def __repr__(self):
        return self._data.__repr__()
    
    def __getitem__(self,x):
        return self._data[x]
    def __eq__(self, other):
        return set(self) == set(other)
    
INSTALLED_PACKAGES = {
    pkg.key for pkg in pkg_resources.working_set
}  # This needs to change since pkg_resources is depricated
BASE_DIR = os.getcwd()

def to_cool_comment(comment: str,padding: tuple[int,int] = (1,1)):
    data = f"{'#' * padding[0]} {comment} {'#' * padding[0]}"
    data_len = len(data)
    vertical_padding = '\n'.join(['#' * data_len for x in range(padding[1])])
    return f"\n\n{vertical_padding}\n{data}\n{vertical_padding}\n\n"

def is_class(node: ast.AST):
        return isinstance(node, ast.ClassDef)


def is_function(node: ast.AST):
    return isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef)


def is_call(node: ast.AST):
    return isinstance(node, ast.Call)


def is_call(node: ast.AST):
    return isinstance(node, ast.Call)


def get_nested_name(start: Union[ast.Attribute, ast.Name, ast.Call]):
        if isinstance(start, ast.Name):
            return start.id
        elif isinstance(start, ast.Attribute):
            return get_nested_name(start.value)
        elif isinstance(start, ast.Call):
            get_nested_name(start.func)
        elif isinstance(start, ast.BinOp):
            return None
        else:
            # print("UNKNOWN NAME",start,start.__dict__)
            pass

        return None


def get_call_name(node: ast.Call) -> str:
    return get_nested_name(node.func)


def get_class_name(node: ast.ClassDef):
    return node.name


def get_function_name(node: ast.FunctionDef):
    return node.name


def get_name_for(node: ast.AST):
    if is_class(node):
        return get_class_name(node)

    if is_function(node):
        return get_function_name(node)

    if is_call(node):
        return get_call_name(node)

    if isinstance(node, ast.Assign):
        return get_nested_name(node.targets[0])

    if isinstance(node, ast.Attribute):
        return get_nested_name(node)

    # print(node,node.__dict__)
    return None

class ReferencesVisitor(ast.NodeVisitor):
    def __init__(
        self,
        deps_for: str = "",
        relevant_refs: list[str] = [],
        checked: OrderedSet[str] = OrderedSet(),
    ):
        self.deps_for = deps_for
        self.relevant_refs = relevant_refs
        self.checked = checked

    def visit_ClassDef(self, node: ClassDef) -> Any:
        self.relevant_refs.append(node.name)
        for x in node.bases:
            self.relevant_refs.append(get_nested_name(x))

        return self.generic_visit(node)

    def visit_FunctionDef(self, node: FunctionDef) -> Any:
        self.relevant_refs.append(node.name)
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: AnnAssign) -> Any:
        if isinstance(node.annotation, ast.Name):
            self.relevant_refs.append(node.annotation.id)
        elif isinstance(node.annotation, ast.Attribute):
            item_name = get_nested_name(node.annotation)
            if item_name is not None and item_name != "self":
                self.relevant_refs.append(item_name)

        return self.generic_visit(node)

    def visit_Name(self, node: Name) -> Any:
        self.relevant_refs.append(get_nested_name(node))

        return self.generic_visit(node)

    def visit_Call(self, node: Call) -> Any:
        item_name = get_call_name(node)
        if item_name is not None and item_name != "self":
            self.relevant_refs.append(item_name)
        return self.generic_visit(node)

    def visit_Attribute(self, node: Attribute) -> Any:
        item_name = get_nested_name(node)
        if item_name is not None and item_name != "self":
            self.relevant_refs.append(item_name)
        return self.generic_visit(node)

    def visit_Expr(self, node: Expr) -> Any:
        item_name = get_nested_name(node.value)
        if item_name is not None and item_name != "self":
            self.relevant_refs.append(item_name)
        return self.generic_visit(node)


class ImportsVisitor(ast.NodeVisitor):
    relativity_regex = r"(?:from|import)(?:[\s]+)?(\.\.|\.|)?([\w\d_\.]+).*"

    def __init__(self, file: str):
        self.file = file
        self.imports_parsed = OrderedDict()

    def visit_Import(self, node):
        import_str = ast.get_source_segment(self.file, node)

        match = re.search(ImportsVisitor.relativity_regex, import_str)

        if match is None:
            self.generic_visit(node)
            return

        relativity, module_name = match[1], match[2]
        module = f"{relativity}{module_name}"

        for alias in node.names:
            # print(alias.__dict__)
            self.imports_parsed[alias.name] = (module, import_str, alias.name)
            if alias.asname is not None:
                self.imports_parsed[alias.asname] = (module, import_str, alias.name)

        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        import_str = ast.get_source_segment(self.file, node)

        match = re.search(ImportsVisitor.relativity_regex, import_str)

        if match is None:
            self.generic_visit(node)
            return

        relativity, module_name = match[1], match[2]
        module = f"{relativity}{module_name}"

        for alias in node.names:
            self.imports_parsed[alias.name] = (module, import_str, alias.name)
            if alias.asname is not None:
                self.imports_parsed[alias.asname] = (module, import_str, alias.name)

        self.generic_visit(node)

class ExtractedRawInfo:
        def __init__(
            self,
            filename: str,
            imports: list[str],
            file_depencencies: dict[str, OrderedSet[str]],
            content: list[tuple[str, str, int]],
        ) -> None:
            #print("FILE",filename,file_depencencies)
            self.filename = filename
            self.imports = imports
            self.file_dependencies = file_depencencies
            self.content = content

        def merge(self, other: "ExtractedRawInfo"):
            self.imports = OrderedSet(self.imports)
            self.imports.update(other.imports)
            self.imports = list(self.imports)

            self.file_dependencies.update(other.file_dependencies)

            self.content = OrderedSet(self.content)
            self.content.update(other.content)
            self.content = list(self.content)


class Extractor:
    def __init__(self) -> None:
        self.paths_cache: dict[str, str] = {}
        self.base_dir = os.getcwd()
        

    def __call__(self, filename: str, names: list[str]) -> Any:
        return self.extract_from_file(filename,names)

    def parse_file_imports(self,file: str, node: ast.AST) -> OrderedDict[str, str]:
        vis = ImportsVisitor(file=file)
        vis.visit(node)
        return vis.imports_parsed


    def get_target_refs(self,tree: ast.AST, names: list[str]) -> OrderedDict[str, ast.AST]:
        deps = OrderedDict()
        for node in tree.body:
            # if isinstance(node,ast.Assign):
            #     node_name = get_name_for(node.value)

            #     if node_name is not None and node_name in names:

            #         deps[node_name] = node
            node_name = get_name_for(node)
            # if (isinstance(node,ast.ClassDef) or isinstance(node,ast.FunctionDef) or isinstance(node,ast.AsyncFunctionDef)) and node.name in names:
            #     deps[node.name] = node

            if node_name is not None and node_name in names:
                deps[node_name] = node

        return deps


    def get_other_refs(self,tree: ast.AST, exclude: list[str]) -> OrderedDict[str, ast.AST]:
        deps = OrderedDict()
        for node in tree.body:
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                continue

            #if isinstance(node, ast.Constant):
                #print(node.__dict__)
                # node_name = get_name_for(node.value)

                # if node_name is not None and node_name not in exclude:

                #     deps[node_name] = node

            node_name = get_name_for(node)
            # if (isinstance(node,ast.ClassDef) or isinstance(node,ast.FunctionDef) or isinstance(node,ast.AsyncFunctionDef)) and node.name not in exclude:
            #     deps[node.name] = node

            if node_name is not None and node_name not in exclude:
                deps[node_name] = node

        return deps


    def module_to_file_path(self,cur_file_path: str, module: str) -> str:
        global INSTALLED_PACKAGES

        cache_key = module

        if cache_key in self.paths_cache:
            return self.paths_cache[cache_key]

        start_path = self.base_dir

        if module.split(".")[0] in INSTALLED_PACKAGES:
            return None

        if module.startswith(".."):
            start_path = os.path.join(os.path.dirname(cur_file_path), "..")
            module = module[2:]
        elif module.startswith("."):
            start_path = os.path.join(os.path.dirname(cur_file_path), ".")
            module = module[1:]

        parts = module.split(".")

        search_path = os.path.abspath(
            os.path.join(start_path, f"{os.path.sep}".join(parts))
        )

        # print(search_path + ".py")
        if os.path.exists(search_path + ".py"):
            self.paths_cache[cache_key] = search_path + ".py"
            return self.paths_cache[cache_key]

        # print(os.path.join(search_path,"__init__.py"))
        if os.path.exists(os.path.join(search_path, "__init__.py")):
            self.paths_cache[cache_key] = os.path.join(search_path, "__init__.py")
            return self.paths_cache[cache_key]

        # print("I give up",module)
        self.paths_cache[cache_key] = None
        return None


    def get_refs_for_root_item(
        self,
        item: ast.AST,
        other_root_refs: OrderedDict[str, ast.AST],
        checked: OrderedSet[str] = OrderedSet(),
        other_root_checked: OrderedSet[str] = OrderedSet(),
    ) -> list[str]:
        # if get_name_for(item) in checked:
        #     return []

        my_refs = []
        visitor = ReferencesVisitor(checked=checked, relevant_refs=my_refs)
        visitor.visit(item)
        checked.add(get_name_for(item))

        for ref in my_refs:
            if ref in other_root_refs.keys() and ref not in checked:
                my_refs.extend(
                    self.get_refs_for_root_item(
                        other_root_refs[ref],
                        other_root_refs=other_root_refs,
                        checked=checked,
                        other_root_checked=other_root_checked,
                    )
                )
        return my_refs

    def extract_raw_info_from_file(
            self,
        filename: str, names: list[str], files_checked_for_refs: OrderedSet[str] = OrderedSet()
    ) -> list[ExtractedRawInfo]:
        for name in names.copy():
            if f"{filename}=>{name}" in files_checked_for_refs:
                names.remove(name)
                

        for name in names:
            files_checked_for_refs.add(f"{filename}=>{name}")

    
        with open(filename, "r", encoding="utf8") as f:
            file = f.read()

        refs_checked_in_this_file: OrderedSet[str] = OrderedSet()

        refs_extracted: OrderedSet[tuple[str, str, int]] = OrderedSet()

        tree = ast.parse(file)

        parsed_imports = self.parse_file_imports(file, tree)

        root_refs = self.get_target_refs(tree=tree, names=names)

        
        other_refs = self.get_other_refs(tree=tree, exclude=names)

        imported_refs = OrderedSet(parsed_imports.keys())

        relevant_refs = []
        others_checked = OrderedSet()

        # Explore functions and classes with no indentation
        for root_ref in root_refs.values():
            relevant_refs.extend(
                self.get_refs_for_root_item(
                    root_ref,
                    other_root_refs=other_refs,
                    checked=refs_checked_in_this_file,
                    other_root_checked=others_checked,
                )
            )
            refs_extracted.add((ast.get_source_segment(file, root_ref), root_ref.lineno))

        relevant_refs = OrderedSet(filter(lambda a: a is not None, relevant_refs))

        # If we find it locally it is likely not an external package
        imported_refs_from_local_files = OrderedSet(
            filter(
                lambda a: a in relevant_refs
                and self.module_to_file_path(filename, parsed_imports[a][0]) is not None,
                imported_refs,
            )
        )

        # If we cannot find it locally then it is likely an external package
        imported_refs_installed = OrderedSet(
            filter(
                lambda a: a in relevant_refs
                and self.module_to_file_path(filename, parsed_imports[a][0]) is None,
                imported_refs,
            )
        )

        # Explore the refs we collected earlier
        for ref in relevant_refs:
            if ref in other_refs.keys():
                ref_node = other_refs.get(ref, None)
                if ref_node is not None:
                    refs_extracted.add(
                        (ast.get_source_segment(file, ref_node), ref_node.lineno)
                    )

        relevant_refs.difference_update(root_refs.keys())
        
        # The refs that exist in other files
        relevant_refs_from_file_imports = relevant_refs.intersection(
            imported_refs_from_local_files
        )

        files_to_get_refs_from: OrderedDict[str, OrderedSet[str]] = OrderedDict()

        # note what files we need to check next
        for ref in relevant_refs_from_file_imports:
            if parsed_imports[ref][2] != ref:
                refs_extracted.add((f"{ref} = {parsed_imports[ref][2]}", -1))
            module_path = self.module_to_file_path(filename, parsed_imports[ref][0])
            if module_path not in files_to_get_refs_from.keys():
                files_to_get_refs_from[module_path] = OrderedSet()

            files_to_get_refs_from[module_path].add(parsed_imports[ref][2])

        # Generate info about this extraction
        all_extracted = [
            ExtractedRawInfo(
                filename=filename,
                imports=list(
                    map(lambda a: parsed_imports.get(a)[1], imported_refs_installed)
                ),
                file_depencencies=files_to_get_refs_from,
                content=list(refs_extracted),
            )
        ]

        # Extract info from other files
        for module_path in files_to_get_refs_from.keys():
            #print("Extracting", files_to_get_refs_from[module_path], "From", module_path)
            all_extracted.extend(
                self.extract_raw_info_from_file(
                    module_path,
                    names=list(files_to_get_refs_from[module_path]),
                    files_checked_for_refs=files_checked_for_refs,
                )
            )

        # Return all extraction results
        return all_extracted


    def extract_from_file(self,filename: str, names: list[str]):
        filename = os.path.abspath(filename)
        extracted = self.extract_raw_info_from_file(filename=filename, names=names)
        collated_extracted: OrderedDict[str, ExtractedRawInfo] = OrderedDict()

        for info in extracted:
            if info.filename not in collated_extracted.keys():
                collated_extracted[info.filename] = info
            else:
                collated_extracted[info.filename].merge(info)


        def comp_a_b(a: str, b: str):
            a_data = collated_extracted[a]
            b_data = collated_extracted[b]

            if a == filename:
                return 1

            if b == filename:
                return -1
            
            a_deps = a_data.file_dependencies.keys()
            b_deps = b_data.file_dependencies.keys()

            if a in b_deps:
                return -1

            if b in a_deps:
                return 1
            
            return len(a_deps) - len(b_deps)

        collated_keys = list(collated_extracted.keys())

        collated_keys.sort(key=cmp_to_key(comp_a_b))


        for x in collated_extracted:
            col_info = collated_extracted[x]
            col_info.content.sort(key=lambda a: a[1])
            # print(collated_extracted[x][2])

        import_parts: OrderedSet() = OrderedSet()
        file_parts = []

        for key in collated_keys:
            col_info = collated_extracted[key]
            file_parts.append(to_cool_comment(f"FILE => {key}"))
            for content, line_no in col_info.content:
                
                file_parts.append(content)
            import_parts.update(col_info.imports)

        return [
            f"# This file was created using https://github.com/TareHimself/python-extractor created by https://github.com/TareHimself\n"
        ] + list(import_parts) + [""], file_parts

class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Extractor",
        description="Extracts symbols from a python file along with their dependencies",
        formatter_class=SmartFormatter,
        exit_on_error=True,
    )

    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="The python file to begin the extraction from",
        required=True,
    )

    parser.add_argument(
        "-s",
        "--symbols",
        nargs="+",
        help="A list of symbols to extract",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output",
        default="out.py",
        type=str,
        help="The name of the output file to place the extracted symbols",
        required=False,
    )

    args = parser.parse_args()

    with open(args.output, "w", encoding="utf8") as out_file:
        e = Extractor()

        file_imports, file_content = e(
            filename=os.path.abspath(args.file), names=args.symbols
        )
        
        out_file.write("\n".join(file_imports + file_content))
