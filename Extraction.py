# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py:percent
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

# %% [markdown]
# ## Subset of full dataset
#
# This notebook extracts 100000 random tweets from the original dataset and saves as a csv

# %%
import pandas as pd
import numpy as np

# %%
import zipfile

# %%
with zipfile.ZipFile("2477_4140_bundle_archive.zip","r") as zip_ref:
    listOfiles = zip_ref.namelist()
    # Iterate over the list of file names in given list & print them
    for elem in listOfiles:
        print(elem)
    zip_ref.extractall("data")

# %%
df = pd.read_csv('data/training.1600000.processed.noemoticon.csv', 
                 encoding =  "ISO-8859-1", 
                names = ['target','id', 'date', 'flag', 'user','text' ])

# %%
df.head()

# %%
df100 = df.sample(n=100000, random_state = 34 )

# %%
len(df['user'].unique()), len(df100['user'].unique())

# %%
len(df['user'].unique()) / len(df), len(df100['user'].unique())/len(df100)

# %%
len(df), len(df100)

# %%
df100.to_csv('data/tweets_100thou.csv', index=False)

# %%
