---
jupyter:
  jupytext:
    formats: ipynb,md,py:percent
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

## Subset of full dataset

This notebook extracts 100000 random tweets from the original dataset and saves as a csv

```python
import pandas as pd
import numpy as np
```

```python
import zipfile
```

```python
with zipfile.ZipFile("2477_4140_bundle_archive.zip","r") as zip_ref:
    listOfiles = zip_ref.namelist()
    # Iterate over the list of file names in given list & print them
    for elem in listOfiles:
        print(elem)
    zip_ref.extractall("data")
```

```python
df = pd.read_csv('data/training.1600000.processed.noemoticon.csv', 
                 encoding =  "ISO-8859-1", 
                names = ['target','id', 'date', 'flag', 'user','text' ])
```

```python
df.head()
```

```python
df100 = df.sample(n=100000, random_state = 34 )
```

```python
len(df['user'].unique()), len(df100['user'].unique())
```

```python
len(df['user'].unique()) / len(df), len(df100['user'].unique())/len(df100)
```

```python
len(df), len(df100)
```

```python
df100.to_csv('data/tweets_100thou.csv', index=False)
```

```python

```
