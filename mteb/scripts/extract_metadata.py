# Use: python extract.py mteb/tasks
# Extracts the MTEB task and dataset metadata

import os
import json
import ast

def find_python_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                yield os.path.join(root, file)

def parse_globals(file_content):
    globals_dict = {}
    try:
        tree = ast.parse(file_content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        try:
                            globals_dict[target.id] = eval(compile(ast.Expression(node.value), '', 'eval'), {})
                        except Exception as e:
                            pass  # Skip if unable to evaluate
    except SyntaxError as e:
        pass  # Ignore files with syntax errors
    return globals_dict

def process_class_node(class_node, globals_dict):
    for node in ast.walk(class_node):
        if isinstance(node, ast.FunctionDef) and node.name == 'description':
            for return_node in ast.walk(node):
                if isinstance(return_node, ast.Return):
                    try:
                        description_dict = eval(compile(ast.Expression(return_node.value), '', 'eval'), globals_dict)
                        eval_langs = description_dict.get("eval_langs")
                        if "en" in eval_langs:
                            return description_dict
                    except Exception as e:
                        pass  # Handle errors silently
    return None

def append_to_json(output_file, data):
    with open(output_file, 'a') as f:
        json.dump(data, f)
        f.write('\n')

def main(directory, output_file):
    folder_stats = {}  # Dictionary to hold the count of instances per folder

    for file_path in find_python_files(directory):
        parent_folder = os.path.basename(os.path.dirname(file_path))
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        globals_dict = parse_globals(file_content)
        tree = ast.parse(file_content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                description = process_class_node(node, globals_dict)
                if description:
                    # Update stats
                    folder_stats[parent_folder] = folder_stats.get(parent_folder, 0) + 1
                    append_to_json(output_file, {
                        "parent_folder": parent_folder,
                        "class_name": node.name,
                        "description": description
                    })

    # Print the stats after processing all files
    for folder, count in folder_stats.items():
        print(f"{folder}: {count} instances")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <directory> <output_file>")
    else:
        directory = sys.argv[1]
        output_file = sys.argv[2]
        main(directory, output_file)
