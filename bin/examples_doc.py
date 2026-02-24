#!/usr/bin/env python3
import os
import re
import shutil


def _extract_order(readme_path):
    if not os.path.exists(readme_path):
        return []
    text = open(readme_path, "r", encoding="utf-8").read()
    return re.findall(r"mcel_zuluaga-[0-9A-Za-z_\-]+\.ipynb", text)


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(root_dir, "ejemplos", "cuadernos-libro")
    docs_examples_dir = os.path.join(root_dir, "docs", "examples")
    rst_file = os.path.join(root_dir, "docs", "examples.rst")
    readme_path = os.path.join(src_dir, "README.md")

    if os.path.exists(docs_examples_dir):
        shutil.rmtree(docs_examples_dir)
    os.makedirs(docs_examples_dir)

    order = _extract_order(readme_path)
    notebook_map = {
        name: os.path.join(src_dir, name)
        for name in os.listdir(src_dir)
        if name.endswith(".ipynb")
    }

    ordered_files = []
    for name in order:
        path = notebook_map.pop(name, None)
        if path:
            ordered_files.append(path)

    for name in sorted(notebook_map.keys()):
        ordered_files.append(notebook_map[name])

    notebooks = []
    for src in ordered_files:
        basename = os.path.basename(src)
        dest = os.path.join(docs_examples_dir, basename)
        shutil.copy2(src, dest)
        notebooks.append(basename)
        print(f"Copied {basename} to docs/examples/")

    with open(rst_file, "w", encoding="utf-8") as rst:
        rst.write("Tutoriales y ejemplos\n")
        rst.write("=====================\n\n")
        rst.write(".. toctree::\n")
        rst.write("   :maxdepth: 2\n")
        rst.write("   :caption: Ejemplos\n\n")

        for nb in notebooks:
            name = os.path.splitext(nb)[0]
            rst.write(f"   examples/{name}\n")

    print("Done.")


if __name__ == "__main__":
    main()
