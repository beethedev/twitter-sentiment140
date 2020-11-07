# ---
# jupyter:
#   jupytext:
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
# # Tweet Sentiment Analysis

# %% [markdown]
# ## Import Statements

# %%
from fastai2.text.all import *
# import plotly.express as px

# %%
df = pd.read_csv('tweets_500thou.csv')

# %%
df.columns

# %% [markdown] heading_collapsed=true
# ## Utilizing out of the box language model from `fastai`

# %% [markdown] hidden=true
# The standard `fastai` language model was build from all the non-trivial Wikipedia articles.

# %% hidden=true
dls = TextDataLoaders.from_df(df, text_col='text', label_col='target', 
                              shuffle_train=False, bs=128)

# %% hidden=true
dls.show_batch()

# %% hidden=true
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

# %% hidden=true
learn.lr_find()

# %% hidden=true
learn.fine_tune(3, 4.4e-2)

# %% hidden=true
learn.save('3epoch_fine_tune_AWD_LSTM')

# %% hidden=true
learn.show_results()

# %% [markdown]
# ## Fine tuning & using custom language model

# %% [markdown]
# ### Custom language model

# %%
dls_lm = TextDataLoaders.from_df(df, is_lm=True, text_col='text', valid_pct=0.2, bs=128, seed=101)

# %%
dls_lm.show_batch(max_n=3)

# %%
learn = language_model_learner(dls_lm, AWD_LSTM, metrics=[accuracy, Perplexity()], drop_mult=0.5)

# %%
learn.lr_find()

# %% [markdown]
# #### 3 epoch Fit One Cycle

# %%
learn.fit_one_cycle(3, 5.2e-2)

# %%
learn.save('3epoch_fit_one_cycle')

# %%
learn.fit_one_cycle(1, 5.2e-2)

# %%
learn = learn.load('3epoch_fit_one_cycle')

# %% [markdown]
# #### 6 epoch Fit One Cycle

# %%
learn = language_model_learner(dls_lm, AWD_LSTM, metrics=[accuracy, Perplexity()], drop_mult=0.5)
learn.fit_one_cycle(6, 6.3e-2)

# %%
learn.save('6epoch_fit_one_cycle')

# %%
learn = learn.load('6epoch_fit_one_cycle')

# %% [markdown]
# #### Fine-tuned with 2-epoch Fit One Cycle

# %%
learn = learn.load('3epoch_fit_one_cycle')

# %%
learn.unfreeze()
learn.fit_one_cycle(2, 6.3e-3)

# %%
learn.save('finetuned_2epoch')

# %%
learn.save_encoder('finetuned_2epoch_encoder')

# %% [markdown]
# It looks like we start to overfit at 3 epochs so let's load the saved encoder from above

# %% [markdown]
# ## Testing Language model

# %%
TEXT = "WALKING HOME"
N_WORDS = 13
N_SENTENCES = 3
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75) 
         for _ in range(N_SENTENCES)]

# %%
print("\n".join(preds))

# %% [markdown]
# ## Train the text classifier (custom language model)

# %%
dls_clas = TextDataLoaders.from_df(df, text_col='text', label_col='target', text_vocab=dls_lm.vocab, seed=102)

# %%
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

# %%
learn = learn.load_encoder('finetuned_2epoch_encoder')

# %%
lrs = learn.lr_find()

# %%
lrs

# %%
lr = lrs[0]  #lr_min

# %%
learn.fit_one_cycle(4, lr)

# %%
learn.save('4epoch_model_class_model')

# %%
learn = learn.load('4epoch_model_class_model')

# %%
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(lr/500,lr/10))

# %%
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(lr/5000,lr/100))

# %%
learn.unfreeze()
learn.fit_one_cycle(2, slice(lr/50000,lr/5000))

# %%
learn.show_results()

# %%
learn.save('final_classifier')

# %%
learn.export('final_classifier.pkl')

# %%
learn = learn.load('final_classifier')

# %% [markdown]
# ## Making and evaluating predictions

# %%
preds = learn.get_preds()

# %%
??learn.get_preds()

# %%
preds[0][:4]

# %%
interp = ClassificationInterpretation.from_learner(learn)

# %%
interp.plot_confusion_matrix()

# %%
interp.plot_confusion_matrix(normalize='true')

# %%
losses = interp.top_losses()

# %%
len(losses[1])

# %%
df_toplosses = interp.plot_top_losses(200)
