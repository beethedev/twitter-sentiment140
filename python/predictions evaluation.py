# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all,-language_info
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
from fastai2.text.all import *

# %%
df = pd.read_csv('tweets_100thou.csv')

# %% [markdown]
# ## Create learner and load saved model

# %%
dls_lm = TextDataLoaders.from_df(df, is_lm=True, text_col='text', valid_pct=0.2, bs=128, shuffle_train=False)

# %%
dls_clas = TextDataLoaders.from_df(df, valid_col="is_valid", text_col='text', label_col='target',shuffle_train=False, text_vocab=dls_lm.vocab)

# %%
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

# %%
learn = learn.load('classifier_81')

# %% [markdown]
# ## Getting predictions

# %%
preds = learn.get_preds()

# %% scrolled=true
valid2 = df[df['is_valid'] == True ]
valid2 = valid2.reset_index()

# %%
df_losses = pd.DataFrame({'loss': losses[0]}, index = losses[1])
df_losses.head()

# %%
df_losses.sort_index()

# %%
df = pd.DataFrame.join(valid2, df_losses)

# %% scrolled=true
df.head()

# %% scrolled=true
interp = ClassificationInterpretation.from_learner(learn)

# %%
losses = interp.top_losses()

# %%

# %%
df_losses.index

# %%
valid.index

# %% scrolled=true
df_toplosses = interp.plot_top_losses(10)

# %%
valid2 = df[df['is_valid'] == True ]

# %%
valid.head()

# %%
valid

# %%
type(preds), type(df_losses), type(df_toplosses)

# %%
top_200losses = df_losses[0:200]

# %%
?? ClassificationInterpretation.plot_top_losses

# %%
?? learn.get_preds

# %%
?? learn.summary

# %%
