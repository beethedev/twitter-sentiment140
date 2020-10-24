# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from jupytext.config import global_jupytext_configuration_directories
list(global_jupytext_configuration_directories())

# %%
from jupytext.config import find_jupytext_configuration_file
find_jupytext_configuration_file('.')

# %%
