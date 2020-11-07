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

# %%
from fastai2.text.all import *
import plotly.express as px

# %%
df = pd.read_csv('tweets_100thou.csv')

# %% [markdown]
# ## Trial 1

# %%
dls = TextDataLoaders.from_df(df, valid_col="is_valid", text_col='text', label_col='target', shuffle_train=False)

# %%
dls.show_batch()

# %%
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

# %%
learn.lr_find()

# %%
learn.fine_tune(4, 1e-3)

# %%
learn.show_results()

# %%
learn.predict("Rest in power, Chadwick. This is all still too difficult to process, thank you for all the movies and all the inspiration. We love you so much")

# %%
learn.predict("Rest in power, chadwick. This is all still too difficult to process")

# %% [markdown]
# ## Fine tuning

# %%
dls_lm = TextDataLoaders.from_df(df, is_lm=True, text_col='text', valid_pct=0.2, bs=128)

# %%
dls_lm.show_batch(max_n=3)

# %%
learn = language_model_learner(dls_lm, AWD_LSTM, metrics=[accuracy, Perplexity()], wd=0.1, )

# %%
learn.lr_find()

# %%
learn.fit_one_cycle(1, 5e-2)

# %%
learn.save('1epoch_fit_one_cycle')

# %%
learn = learn.load('1epoch_fit_one_cycle')

# %%
learn.unfreeze()
learn.fit_one_cycle(6, 5e-3)

# %%
learn.save_encoder('finetuned')

# %%
TEXT = "WALKING HOME"
N_WORDS = 13
N_SENTENCES = 3
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75) 
         for _ in range(N_SENTENCES)]

# %%
print("\n".join(preds))

# %% [markdown]
# ## Train the text classifier

# %%
dls_clas = TextDataLoaders.from_df(df, valid_col="is_valid", text_col='text', label_col='target', text_vocab=dls_lm.vocab)

# %%
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

# %%
learn = learn.load_encoder('finetuned')

# %%
learn.lr_find()

# %%
learn.fit_one_cycle(1, 1e-2)

# %%
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))

# %%
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(1e-4/(2.6**4),1e-4))

# %%
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4)/5,1e-3/5))

# %%
learn.fit_one_cycle(2, slice(1e-3/(2.6**4)/5,1e-3/5))

# %%
learn.show_results()

# %%
learn.fit_one_cycle(1, slice(1e-4/(2.6**4),1e-4))

# %%
learn.fit_one_cycle(1, slice(1e-4/(2.6**4),1e-4))

# %%
learn.save('classifier_81')

# %%
learn.export('classifier_81.pkl')

# %%
learn = learn.load('classifier_81')

# %% [markdown]
# ## Making and evaluating predictions

# %%
preds = learn.get_preds()

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
df_losses = pd.DataFrame({'loss': losses[0],
                         'index': losses[1]})

# %%
df_losses.head()

# %%
df_toplosses = interp.plot_top_losses(200)

# %%
top_200losses = df_losses[0:200]

# %%
top_200losses.merge(df_toplosses, on='loss')

# %%

# %%
valid = dls_clas.valid.items

# %%
valid = valid[['text', 'is_valid', 'target']]

# %%
valid.head()

# %%
valid[l] = valid.assign(prediction=pd.Series(preds[1]))

# %%
valid = valid.assign(prediction=pd.Series(preds[1]))

# %%
valid = valid.assign(probability_0=Series(preds[0]))

# %%
valid = valid.assign(probability_0=Series(preds[0]))

# %%
max(preds[0][0])

# %%
preds[0][0]

# %%
most_confused = interp.most_confused()

# %%
most_confused #actual, predicted, no of ocurences

# %%
report = interp.print_classification_report()

# %%
?? ClassificationInterpretation.plot_top_losses

# %%
