---
jupyter:
  jupytext:
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

# Tweet Sentiment Analysis


## Import Statements

```python
from fastai2.text.all import *
# import plotly.express as px
```

```python
df = pd.read_csv('tweets_500thou.csv')
```

```python
df.columns
```

<!-- #region heading_collapsed=true -->
## Utilizing out of the box language model from `fastai`
<!-- #endregion -->

<!-- #region hidden=true -->
The standard `fastai` language model was build from all the non-trivial Wikipedia articles.
<!-- #endregion -->

```python hidden=true
dls = TextDataLoaders.from_df(df, text_col='text', label_col='target', 
                              shuffle_train=False, bs=128)
```

```python hidden=true
dls.show_batch()
```

```python hidden=true
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
```

```python hidden=true
learn.lr_find()
```

```python hidden=true
learn.fine_tune(3, 4.4e-2)
```

```python hidden=true
learn.save('3epoch_fine_tune_AWD_LSTM')
```

```python hidden=true
learn.show_results()
```

## Fine tuning & using custom language model


### Custom language model

```python
dls_lm = TextDataLoaders.from_df(df, is_lm=True, text_col='text', valid_pct=0.2, bs=128, seed=101)
```

```python
dls_lm.show_batch(max_n=3)
```

```python
learn = language_model_learner(dls_lm, AWD_LSTM, metrics=[accuracy, Perplexity()], drop_mult=0.5)
```

```python
learn.lr_find()
```

#### 3 epoch Fit One Cycle

```python
learn.fit_one_cycle(3, 5.2e-2)
```

```python
learn.save('3epoch_fit_one_cycle')
```

```python
learn.fit_one_cycle(1, 5.2e-2)
```

```python
learn = learn.load('3epoch_fit_one_cycle')
```

#### 6 epoch Fit One Cycle

```python
learn = language_model_learner(dls_lm, AWD_LSTM, metrics=[accuracy, Perplexity()], drop_mult=0.5)
learn.fit_one_cycle(6, 6.3e-2)
```

```python
learn.save('6epoch_fit_one_cycle')
```

```python
learn = learn.load('6epoch_fit_one_cycle')
```

#### Fine-tuned with 2-epoch Fit One Cycle

```python
learn = learn.load('3epoch_fit_one_cycle')
```

```python
learn.unfreeze()
learn.fit_one_cycle(2, 6.3e-3)
```

```python
learn.save('finetuned_2epoch')
```

```python
learn.save_encoder('finetuned_2epoch_encoder')
```

It looks like we start to overfit at 3 epochs so let's load the saved encoder from above


## Testing Language model

```python
TEXT = "WALKING HOME"
N_WORDS = 13
N_SENTENCES = 3
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75) 
         for _ in range(N_SENTENCES)]
```

```python
print("\n".join(preds))
```

## Train the text classifier (custom language model)

```python
dls_clas = TextDataLoaders.from_df(df, text_col='text', label_col='target', text_vocab=dls_lm.vocab, seed=102)
```

```python
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
```

```python
learn = learn.load_encoder('finetuned_2epoch_encoder')
```

```python
lrs = learn.lr_find()
```

```python
lrs
```

```python
lr = lrs[0]  #lr_min
```

```python
learn.fit_one_cycle(4, lr)
```

```python
learn.save('4epoch_model_class_model')
```

```python
learn = learn.load('4epoch_model_class_model')
```

```python
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(lr/500,lr/10))
```

```python
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(lr/5000,lr/100))
```

```python
learn.unfreeze()
learn.fit_one_cycle(2, slice(lr/50000,lr/5000))
```

```python
learn.show_results()
```

```python
learn.save('final_classifier')
```

```python
learn.export('final_classifier.pkl')
```

```python
learn = learn.load('final_classifier')
```

## Making and evaluating predictions

```python
preds = learn.get_preds()
```

```python
??learn.get_preds()
```

```python
preds[0][:4]
```

```python
interp = ClassificationInterpretation.from_learner(learn)
```

```python
interp.plot_confusion_matrix()
```

```python
interp.plot_confusion_matrix(normalize='true')
```

```python
losses = interp.top_losses()
```

```python
len(losses[1])
```

```python
df_toplosses = interp.plot_top_losses(200)
```

```python
top_200losses = df_losses[0:200]
```

```python
top_200losses.merge(df_toplosses, on='loss')
```

```python

```

```python
valid = dls_clas.valid.items
```

```python
valid = valid[['text', 'is_valid', 'target']]
```

```python
valid.head()
```

```python
valid[l] = valid.assign(prediction=pd.Series(preds[1]))
```

```python
valid = valid.assign(prediction=pd.Series(preds[1]))
```

```python
valid = valid.assign(probability_0=Series(preds[0]))
```

```python
valid = valid.assign(probability_0=Series(preds[0]))
```

```python
max(preds[0][0])
```

```python
preds[0][0]
```

```python
most_confused = interp.most_confused()
```

```python
most_confused #actual, predicted, no of ocurences
```

```python
report = interp.print_classification_report()
```

```python
?? ClassificationInterpretation.plot_top_losses
```

```python

```
