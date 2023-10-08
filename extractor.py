from _ast import ClassDef, FunctionDef, Nonlocal
from ast import AnnAssign, Attribute, Call, ClassDef, Expr, FunctionDef, Name, arguments
import ast
from typing import Any
import os
import re
import sys
from importlib.resources import is_resource
from typing import Union
import pkg_resources
from functools import cmp_to_key
INSTALLED_PACKAGES = {pkg.key for pkg in pkg_resources.working_set}
BASE_DIR = os.getcwd()

# This file is an attempt to extract a given function or name from a module with all its depencencies. it has trouble with relative imports and needs more work

class ReferencesVisitor(ast.NodeVisitor):
    def __init__(self,deps_for: str = "",relevant_refs: list[str] = [],checked: set[str] = set()):
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
        if isinstance(node.annotation,ast.Name):
            self.relevant_refs.append(node.annotation.id)
        elif isinstance(node.annotation,ast.Attribute):
            item_name = get_nested_name(node.annotation)
            if item_name is not None and item_name != 'self':
                self.relevant_refs.append(item_name)

        return self.generic_visit(node)
    
    def visit_Name(self, node: Name) -> Any:
        self.relevant_refs.append(get_nested_name(node))

        return self.generic_visit(node)
    
    def visit_Call(self, node: Call) -> Any:
        item_name = get_call_name(node)
        if item_name is not None and item_name != 'self':
            self.relevant_refs.append(item_name)
        return self.generic_visit(node)
    
    def visit_Attribute(self, node: Attribute) -> Any:
        item_name = get_nested_name(node)
        if item_name is not None and item_name != 'self':
            self.relevant_refs.append(item_name)
        return self.generic_visit(node)
    
    
    def visit_Expr(self, node: Expr) -> Any:
        item_name = get_nested_name(node.value)
        if item_name is not None and item_name != 'self':
            self.relevant_refs.append(item_name)
        return self.generic_visit(node)
    
    
class ImportsVisitor(ast.NodeVisitor):
    relativity_regex = r"(?:from|import)(?:[\s]+)?(\.\.|\.|)?([\w\d_\.]+).*"
    def __init__(self,file: str):
        self.file = file
        self.imports_parsed = {}

    def visit_Import(self, node):
        
        import_str = ast.get_source_segment(self.file,node)

        match = re.search(ImportsVisitor.relativity_regex,import_str)
        
        if match is None:
            self.generic_visit(node)
            return
        
        relativity,module_name = match[1],match[2]
        module = f"{relativity}{module_name}"

        for alias in node.names:
            # print(alias.__dict__)
            self.imports_parsed[alias.name] = (module,import_str,alias.name)
            if alias.asname is not None:
                self.imports_parsed[alias.asname] = (module,import_str,alias.name)

        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        import_str = ast.get_source_segment(self.file,node)

        match = re.search(ImportsVisitor.relativity_regex,import_str)
        
        if match is None:
            self.generic_visit(node)
            return
        
        relativity,module_name = match[1],match[2]
        module = f"{relativity}{module_name}"

        for alias in node.names:
            self.imports_parsed[alias.name] = (module,import_str,alias.name)
            if alias.asname is not None:
                self.imports_parsed[alias.asname] = (module,import_str,alias.name)

        self.generic_visit(node) 

def is_class(node: ast.AST):
    return isinstance(node,ast.ClassDef)

def is_function(node: ast.AST):
    return isinstance(node,ast.FunctionDef) or isinstance(node,ast.AsyncFunctionDef)

def is_call(node: ast.AST):
    return isinstance(node,ast.Call)

def is_call(node: ast.AST):
    return isinstance(node,ast.Call)

def get_nested_name(start: Union[ast.Attribute,ast.Name,ast.Call]):
    if isinstance(start,ast.Name):
            return start.id
    elif isinstance(start,ast.Attribute):
        return get_nested_name(start.value)
    elif isinstance(start,ast.Call):
        get_nested_name(start.func)
    elif isinstance(start,ast.BinOp):
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
    
    if isinstance(node,ast.Assign):
        return get_nested_name(node.targets[0])
    
    if isinstance(node,ast.Attribute):
        return get_nested_name(node)
    
    # print(node,node.__dict__)
    return None


def parse_file_imports(file: str,node: ast.AST) -> dict[str,str]:
    vis = ImportsVisitor(file=file)
    vis.visit(node)
    return vis.imports_parsed

def get_target_refs(tree: ast.AST,names: list[str]) -> dict[str,ast.AST]:
    deps = {}
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

def get_other_refs(tree: ast.AST,exclude: list[str]) -> dict[str,ast.AST]:
    deps = {}
    for node in tree.body:
        if isinstance(node,ast.Import) or isinstance(node,ast.ImportFrom):
            continue

        if isinstance(node,ast.Constant):
            print(node.__dict__)
            # node_name = get_name_for(node.value)

            # if node_name is not None and node_name not in exclude:
                
            #     deps[node_name] = node

        node_name = get_name_for(node)
        # if (isinstance(node,ast.ClassDef) or isinstance(node,ast.FunctionDef) or isinstance(node,ast.AsyncFunctionDef)) and node.name not in exclude:
        #     deps[node.name] = node

        if node_name is not None and node_name not in exclude:
            deps[node_name] = node

    return deps
    

paths_cache: dict[str,str] = {}

def module_to_file_path(cur_file_path: str,module: str) -> str:
    global BASE_DIR
    global paths_cache
    global INSTALLED_PACKAGES

    cache_key = module
    if cache_key in paths_cache:
        return paths_cache[cache_key]
    
    start_path = BASE_DIR

    if module.split('.')[0] in INSTALLED_PACKAGES:
        return None
    

    
    if module.startswith('..'):
        start_path = os.path.join(os.path.dirname(cur_file_path),'..')
        module = module[2:]
    elif module.startswith('.'):
        start_path = os.path.join(os.path.dirname(cur_file_path),'.')
        module = module[1:]


    parts = module.split(".")

    search_path = os.path.abspath(os.path.join(start_path,f"{os.path.sep}".join(parts)))

    #print(search_path + ".py")
    if os.path.exists(search_path + ".py"):
        
        paths_cache[cache_key] = search_path + ".py"
        return paths_cache[cache_key]

    #print(os.path.join(search_path,"__init__.py"))
    if os.path.exists(os.path.join(search_path,"__init__.py")):
        paths_cache[cache_key] = os.path.join(search_path,"__init__.py")
        return paths_cache[cache_key]
    
    #print("I give up",module)
    paths_cache[cache_key] = None
    return None


def get_refs_for_root_item(item: ast.AST,other_root_refs: dict[str,ast.AST],checked: set[str] = set(),other_root_checked: set[str] = set()) -> list[str]:
    # if get_name_for(item) in checked:
    #     return []
    
    my_refs = []
    visitor = ReferencesVisitor(checked=checked,relevant_refs=my_refs)
    visitor.visit(item)
    checked.add(get_name_for(item))

    for ref in my_refs:
        if ref in other_root_refs.keys() and ref not in checked:
            my_refs.extend(get_refs_for_root_item(other_root_refs[ref],other_root_refs=other_root_refs,checked=checked,other_root_checked=other_root_checked))
    return my_refs

class ExtractedRawInfo:
    def __init__(self,filename: str,imports: list[str],file_depencencies:  dict[str, set[str]],content: list[tuple[str,str,int]]) -> None:
        self.filename = filename
        self.imports = imports
        self.file_dependencies = file_depencencies
        self.content = content

    def merge(self,other: 'ExtractedRawInfo'):
        self.imports = set(self.imports)
        self.imports.update(other.imports)
        self.imports = list(self.imports)

        self.file_dependencies.update(other.file_dependencies)
        
        self.content = set(self.content)
        self.content.update(other.content)
        self.content = list(self.content)

def extract_raw_info_from_file(filename:str,names: list[str],files_checked_for_refs: set[str] = set()) -> list[ExtractedRawInfo]:
    for name in names.copy():
        if f"{filename}=>{name}" in files_checked_for_refs:
            names.remove(name)

    for name in names:
        files_checked_for_refs.add(f"{filename}=>{name}")

    
    # print("CHECKING FILE",filename,names)
    with open(filename,'r',encoding="utf8") as f:
        file = f.read()

    refs_checked_in_this_file: set[str] = set()

    refs_extracted: set[tuple[str,str,int]] = set()

    tree = ast.parse(file)

    parsed_imports = parse_file_imports(file,tree)

    root_refs = get_target_refs(tree=tree,names=names)

    other_refs = get_other_refs(tree=tree,exclude=names)

    imported_refs = set(parsed_imports.keys())

    
    relevant_refs = []
    others_checked = set()
    for root_ref in root_refs.values():

        relevant_refs.extend(get_refs_for_root_item(root_ref,other_root_refs=other_refs,checked=refs_checked_in_this_file,other_root_checked=others_checked))
        refs_extracted.add((ast.get_source_segment(file,root_ref),root_ref.lineno))

    relevant_refs = set(filter(lambda a: a is not None,relevant_refs))
    
    
    imported_refs_from_local_files = set(filter(lambda a: a in relevant_refs and module_to_file_path(filename,parsed_imports[a][0]) is not None,imported_refs))

    imported_refs_installed = set(filter(lambda a: a in relevant_refs and module_to_file_path(filename,parsed_imports[a][0]) is None,imported_refs))

    for ref in relevant_refs:
        if ref in other_refs.keys():
            ref_node = other_refs.get(ref,None)
            if ref_node is not None:
                refs_extracted.add((ast.get_source_segment(file,ref_node),ref_node.lineno))
    

    relevant_refs.difference_update(root_refs.keys())

    relevant_refs_from_file_imports = relevant_refs.intersection(imported_refs_from_local_files)

    files_to_get_refs_from: dict[str,set[str]] = {}


    for ref in relevant_refs_from_file_imports:
        if parsed_imports[ref][2] != ref:
            refs_extracted.add((f"{ref} = {parsed_imports[ref][2]}",-1))
        module_path = module_to_file_path(filename,parsed_imports[ref][0])
        if module_path not in files_to_get_refs_from.keys():
            files_to_get_refs_from[module_path] = set()

        files_to_get_refs_from[module_path].add(parsed_imports[ref][2])


    all_extracted = [ExtractedRawInfo(filename=filename,imports=list(map(lambda a: parsed_imports.get(a)[1],imported_refs_installed)),file_depencencies=files_to_get_refs_from,content=list(refs_extracted))]

    for module_path in files_to_get_refs_from.keys():
        print("Extracting",files_to_get_refs_from[module_path],"From",module_path)
        all_extracted.extend(extract_raw_info_from_file(module_path,names=list(files_to_get_refs_from[module_path]),files_checked_for_refs=files_checked_for_refs))

    return all_extracted




def extract_from_file(filename:str,names: list[str]):
    filename = os.path.abspath(filename)
    extracted = extract_raw_info_from_file(filename=filename,names=names)
    collated_extracted: dict[str,ExtractedRawInfo] = {}

    for info in extracted:
        if info.filename not in collated_extracted.keys():
            collated_extracted[info.filename] = info
        else:
            collated_extracted[info.filename].merge(info)
    
    def comp_a_b(a:str,b: str):
        a_data = collated_extracted[a]
        b_data = collated_extracted[b]

        if a == filename:
            return -1
        
        if b == filename:
            return 1
        

        if a in b_data.file_dependencies.keys():
            return 1
        
        if b in a_data.file_dependencies.keys():
            return -1
    
        
        return len(a_data.file_dependencies.keys()) - len(b_data.file_dependencies.keys())
    collated_keys = list(collated_extracted.keys())
    
    for x in collated_keys:
        collated_keys.sort(key=cmp_to_key(comp_a_b))

    for x in collated_extracted:
        col_info = collated_extracted[x]
        col_info.content.sort(key=lambda a: a[1])
        # print(collated_extracted[x][2])

    import_parts: set() = set()
    file_parts = []

    for key in reversed(collated_keys):
        col_info = collated_extracted[key]
        #print(key,col_content)
        for content,line_no in col_info.content:
            print(key)
            file_parts.append(content)
        import_parts.update(col_info.imports)
    
    
    return [f"# This file was created using extractor.py\n"] + list(import_parts) + [""],file_parts


with open("out.py",'w',encoding='utf8') as out_file:
    file_imports,file_content = extract_from_file(filename="D:\\Github\\manga-translator\\d.py",names=["non_max_suppression","scale_boxes","process_mask"])
    out_file.write("\n".join(file_imports + file_content))