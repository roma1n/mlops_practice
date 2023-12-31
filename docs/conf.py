# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import sys
import tomllib


with open("../pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MLOps Pracatice"
copyright = "2023, Roman Golubtsov"
author = "Roman Golubtsov"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"

# TODO: Take from pyproject.toml
autodoc_mock_imports = list(
    map(
        lambda module_name: module_name.replace("-", "_"),
        pyproject["tool"]["poetry"]["dependencies"].keys(),
    )
) + [
    "sklearn",
    "hydra",
    "torch",
    "git",
]
