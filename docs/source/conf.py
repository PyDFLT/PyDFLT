import datetime
import os
import pathlib
import shutil
import sys

try:
    import pypandoc
except ImportError:  # optional dependency for notebook conversion
    pypandoc = None

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

EXAMPLES_DIR = pathlib.Path(__file__).parents[2] / "examples"
DOCS_EXAMPLES_DIR = pathlib.Path(__file__).parent / "examples"
DOCS_EXAMPLES_DIR.mkdir(exist_ok=True)
STATIC_DIR = pathlib.Path(__file__).parent / "_static"
STATIC_DIR.mkdir(exist_ok=True)

# keep notebooks in docs/examples in sync with top-level examples
for notebook in DOCS_EXAMPLES_DIR.glob('*.ipynb'):
    notebook.unlink()
for notebook in sorted(EXAMPLES_DIR.glob('*.ipynb')):
    shutil.copy2(notebook, DOCS_EXAMPLES_DIR / notebook.name)

LOGO_SOURCE = pathlib.Path(__file__).parents[2] / 'images' / 'logo_small.png'
if LOGO_SOURCE.exists():
    shutil.copy2(LOGO_SOURCE, STATIC_DIR / 'logo_small.png')

if pypandoc is not None:
    pandoc_path = pathlib.Path(pypandoc.get_pandoc_path())
    os.environ.setdefault('PANDOC', str(pandoc_path))
    os.environ['PATH'] = str(pandoc_path.parent) + os.pathsep + os.environ.get('PATH', '')

# Project information
now = datetime.date.today()

project = "PyDFLT"
authors = "Noah Schutte, Kim van den Houten, Grigorii Veviurko"
copyright = f"2025 - {now.year}, {authors}"


# -- General configuration
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_immaterial",
    "nbsphinx",
    "numpydoc",
]

templates_path = ["_templates"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/.ipynb_checkpoints"]

add_module_names = False
python_use_unqualified_type_names = True

# -- API documentation
autoclass_content = "class"
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
autodoc_preserve_defaults = True

# -- numpydoc
numpydoc_xref_param_type = True
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = False
napoleon_include_special_with_doc = True


# -- intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
intersphinx_disabled_domains = ["std"]

# -- nbsphinx
skip_notebooks_env = os.getenv("SKIP_NOTEBOOKS", "1")
skip_notebooks = skip_notebooks_env not in {"0", "false", "False"}
nbsphinx_execute = "never" if skip_notebooks else "auto"


nbsphinx_markdown_renderer = "myst"


# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_immaterial"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo_small.png"

html_theme_options = {
    "repo_url": "https://github.com/PyDFLT/PyDFLT/",
    "icon": {
        "repo": "fontawesome/brands/github",
        "edit": "material/file-edit-outline",
    },
    "features": [
        "navigation.expand",
        "navigation.top",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "primary": "blue",
            "accent": "green",
            "scheme": "default",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "primary": "blue",
            "accent": "green",
            "scheme": "slate",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to light mode",
            },
        },
    ],
}

object_description_options = [
    (
        "py:.*",
        {"include_fields_in_toc": False, "include_rubrics_in_toc": False},
    ),
    ("py:attribute", {"include_in_toc": False}),
    ("py:parameter", {"include_in_toc": False}),
]
