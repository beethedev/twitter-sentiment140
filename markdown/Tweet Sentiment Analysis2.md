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

```python
from fastai2.text.all import *
import plotly.express as px
```

```python
df = pd.read_csv('tweets_100thou.csv')
```

## Trial 1

```python
dls = TextDataLoaders.from_df(df, valid_col="is_valid", text_col='text', label_col='target', shuffle_train=False)
```

```python
dls.show_batch()
```

```python
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
```

```python
learn.lr_find()
```

```python
learn.fine_tune(4, 1e-3)
```

```python
learn.show_results()
```

```python
learn.predict("Rest in power, Chadwick. This is all still too difficult to process, thank you for all the movies and all the inspiration. We love you so much")
```

```python
learn.predict("Rest in power, chadwick. This is all still too difficult to process")
```

## Fine tuning

```python
dls_lm = TextDataLoaders.from_df(df, is_lm=True, text_col='text', valid_pct=0.2, bs=128)
```

```python
dls_lm.show_batch(max_n=3)
```

```python
learn = language_model_learner(dls_lm, AWD_LSTM, metrics=[accuracy, Perplexity()], wd=0.1, )
```

```python
learn.lr_find()
```

```python
learn.fit_one_cycle(1, 5e-2)
```

```python
learn.save('1epoch_fit_one_cycle')
```

```python
learn = learn.load('1epoch_fit_one_cycle')
```

```python
learn.unfreeze()
learn.fit_one_cycle(6, 5e-3)
```

```python
learn.save_encoder('finetuned')
```

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

## Train the text classifier

```python
dls_clas = TextDataLoaders.from_df(df, valid_col="is_valid", text_col='text', label_col='target', text_vocab=dls_lm.vocab)
```

```python
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
```

```python
learn = learn.load_encoder('finetuned')
```

```python
learn.lr_find()
```

```python
learn.fit_one_cycle(1, 1e-2)
```

```python
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))
```

```python
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(1e-4/(2.6**4),1e-4))
```

```python
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4)/5,1e-3/5))
```

```python
learn.fit_one_cycle(2, slice(1e-3/(2.6**4)/5,1e-3/5))
```

```python
learn.show_results()
```

```python
learn.fit_one_cycle(1, slice(1e-4/(2.6**4),1e-4))
```

```python
learn.fit_one_cycle(1, slice(1e-4/(2.6**4),1e-4))
```

```python
learn.save('classifier_81')
```

```python
learn.export('classifier_81.pkl')
```

```python
learn = learn.load('classifier_81')
```

## Making and evaluating predictions

```python
preds = learn.get_preds()
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
df_losses = pd.DataFrame({'loss': losses[0],
                         'index': losses[1]})
```

```python
df_losses.head()
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
