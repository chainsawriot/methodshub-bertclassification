# Fine-Tuning Transformer Models for Classification of Digital
Behavioural Data
Indira Sen

## Learning Objectives

By the end of this tutorial, you will be able to fine-tune transformer
models like BERT for binary and multiclass document classification. We
show how to use `simpletransformers` for using transformer models in
Python. <a href="#sec-transformers" class="quarto-xref">Section 9.1</a>
shows how to use `HuggingFace` to train the same model.
<a href="#sec-multiclass" class="quarto-xref">Section 10</a> expands the
binary classification to multiclass.

As an example, we will fine-tune a specific transformer model
(DistilBERT) for automatic sexism detection.

## Target audience

This tutorial is aimed at social scientists with some knowledge in
Python and supervised machine learning.

## Setting up the computational environment

The following Python packages are required

``` python
!pip install pandas numpy torch sklearn
!pip install simpletransformers
!pip install transformers[torch]
```

This package is optional

``` python
!pip install accelerate -U
```

## Duration

It depends on the hardware. This notebook can be used with or without
GPU compute, but it’s much faster if you do have a GPU.

## Social Science Usecase(s)

This method has been used in Samory et al. (2021) for automatic sexism
detection.

## Import necessary Python libraries and modules

We will import necessary Python libraries and modules.

``` python
import pandas as pd
import numpy as np

# For deep learning
import torch
```

We then check if have a GPU available. This is important because some
parts of the code have to be modified later on based on this.

``` python
gpu_avail = torch.cuda.is_available()
```

## Get the data

We first download the datasets we need for finetuning our models. This
is a **supervised** classification task, therefore, we will need labeled
data. We download the the ‘Call me sexist but’ (CMSB) dataset which you
can find here: <https://search.gesis.org/research_data/SDN-10.7802-2251>
This dataset is from our paper on detecting sexism in a theory-driven
manner (Samory et al., 2021).

Here, we can download the full data (‘sexism_data.csv’) and put it in
folder called ‘sexism_data’ in the same location as our jupyter
notebook. Here, we will use just a subset of the data
(‘sexism_sample.csv’) for demonstration.

``` python
## sexism_data = pd.read_csv('sexism_data/sexism_data.csv')

sexism_data = pd.read_csv('sexism_data/sexism_sample.csv')

sexism_data.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | id | dataset | text | toxicity | sexist | of_id |
|----|----|----|----|----|----|----|
| 0 | 10330 | other | MENTION3839 MENTION2475 would have to see what... | 0.203880 | False | -1 |
| 1 | 5703 | callme | I hate it when women beat up men in movies! | 0.661064 | True | -1 |
| 2 | 12331 | scales | Men are better at science and math than women are | 0.531301 | True | -1 |
| 3 | 12855 | callme | "MENTION149: MENTION2551 ps I'd never listen t... | 0.278660 | False | -1 |
| 4 | 2240 | other | MENTION279 Vivek W is the Carrot Top of academ... | 0.084303 | False | -1 |

</div>

``` python
sexism_data = sexism_data.dropna(subset = 'sexist')
```

## Finetuning a classifier: the general procedure

### With `simpletransformers`

We first use the [`simpletransformers`](https://simpletransformers.ai/)
package which is more beginner-friendly. The basic steps for finetuning
a classifier using simpletrasnformers are:

- Initialize a model based on a specific architechture (BERT,
  DistilBERT, etc)
- Train the model with train_model()
- Evaluate the model with eval_model()
- Make predictions on (unlabelled) data with predict()

``` python
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
```

``` python
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
```

We need to preprocess the data first before we start the finetuning
process. In this step, we split the dataset into **train** and **test**
sets to have a fully held-out test set that can be used to evaluate our
classifier.

We can also create a **validation** that is used during the fine tuning
process for hyperparameter tuning, but that is not mandatory.

``` python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(sexism_data, stratify=sexism_data['sexist'], test_size=0.2)
```

We now convert the dataframes into a format that can be read by
simpletransformers. This is a dataframe with the columns ‘text’ and
‘labels’. The ‘labels’ column should be numerical, so we use **one-hot
encoding** to transform our boolean sexist labels to numerical ones.

``` python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train_df['sexist'])
train_df['labels'] = le.transform(train_df['sexist'])
test_df['labels'] = le.transform(test_df['sexist'])
```

``` python
# to see which number was mapped to which class:
list(le.inverse_transform([0,1]))
```

    [np.False_, np.True_]

So, 0 is non-sexist and 1 is sexist. We now have the appropriate data
structure.

The next step is setting the training parameters and loading the
classification model, in this case, DistilBERT (Sanh et al., 2019), a
lightweight model that can be trained relatively quickly compared to
other transformer variants like BERT and RoBERTa.

For training parameters, we have many to choose from such as the
learning rate, whether we want to stop early or not, where we should
save the model, and more. You can find all of them
[here](https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model).

As a minimal setup, we will just set the number of **epochs**, i.e., the
number of passes the model does over the full training set. For recent
transformer models, epochs are usually set to 2 or 3, after which
overfitting may happen.

**use_cuda** is a parameter that signals whether the GPU should be used
or not. It will be set based on our check earlier.

``` python
# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=3, overwrite_output_dir=True)

# Create a ClassificationModel
model = ClassificationModel(
    "distilbert", "distilbert-base-uncased", args=model_args, use_cuda=gpu_avail,
)

# we set some additional parameters when using a GPU
if gpu_avail:
    model_args.use_multiprocessing=False
    model_args.use_multiprocessing_for_evaluation=False
```

We are now finally ready to begin training! This might take a while,
especially when we’re not using a GPU.

``` python
# Train the model
model.train_model(train_df)
```

    Epoch:   0%|          | 0/3 [00:00<?, ?it/s]

    Running Epoch 1 of 3:   0%|          | 0/20 [00:00<?, ?it/s]

    Running Epoch 2 of 3:   0%|          | 0/20 [00:00<?, ?it/s]

    Running Epoch 3 of 3:   0%|          | 0/20 [00:00<?, ?it/s]

    (60, 0.5564038594563802)

After training our model, we can use it to make predictions for
unlabeled datapoints to classify whether they are sexist or not.

``` python
sexist_tweet = "A woman will never be truly fulfilled in life if she doesn’t have a committed long-term relationship with a man"
predictions, raw_outputs = model.predict([sexist_tweet])
le.inverse_transform(predictions)
```

      0%|          | 0/1 [00:00<?, ?it/s]

    array([ True])

``` python
nonsexist_tweet = "International Women's Day (IWD) is a holiday celebrated annually on March 8 as a focal point in the women's rights movement."
predictions, raw_outputs = model.predict([nonsexist_tweet])
le.inverse_transform(predictions)
```

      0%|          | 0/1 [00:00<?, ?it/s]

    array([ True])

We can also use the held-out test set to quantitatively evaluate our
model.

``` python
# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(test_df)
result
```

    Running Evaluation:   0%|          | 0/1 [00:00<?, ?it/s]

    {'mcc': np.float64(0.6713171133426189),
     'accuracy': 0.825,
     'f1_score': 0.8444444444444444,
     'tp': np.int64(19),
     'tn': np.int64(14),
     'fp': np.int64(6),
     'fn': np.int64(1),
     'auroc': np.float64(0.9299999999999999),
     'auprc': np.float64(0.9545054047259929),
     'eval_loss': 0.4366455078125}

``` python
# you can also use sklearn's neat classification report to get more metrics
from sklearn.metrics import classification_report

preds, _ = model.predict(list(test_df['text'].values))
# preds = le.inverse_transform(preds)

print(classification_report(test_df['labels'], preds))
```

      0%|          | 0/1 [00:00<?, ?it/s]

                  precision    recall  f1-score   support

               0       0.93      0.70      0.80        20
               1       0.76      0.95      0.84        20

        accuracy                           0.82        40
       macro avg       0.85      0.82      0.82        40
    weighted avg       0.85      0.82      0.82        40

## Conclusion

That’s a wrap on fine-tuning your own transformer models for text
classification. You can replace the sexism dataset with any other
labeled dataset of your choice for a particular task to train a
classifier for that task. More further reading and examples, see:

- <https://www.aiforhumanists.com/tutorials/>
- <https://huggingface.co/docs/transformers/en/training>

### Optional: HuggingFace `transformers`

We now repeat the same process with the HuggingFace [`transformers`
Python library](https://huggingface.co/transformers/installation.html).
Additionally, we also use the [accelerate
library](https://huggingface.co/docs/accelerate/index), which helps make
our code more efficient. We will again use DistilBERT.

``` python
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
```

We will set some of the configurations, including whether to use a GPU
or not.

``` python
model_name = 'distilbert-base-uncased'
if gpu_avail:
    device_name = 'cuda'
else:
    device_name = 'cpu'

# This is the maximum number of tokens in any document; the rest will be truncated.
max_length = 512

# This is the name of the directory where we'll save our model. You can name it whatever you want.
cached_model_directory_name = 'output_hf'
```

We will reuse the train-test splits we created for simpletransformers,
but change the data structure slightly.

``` python
train_texts = train_df['text'].values
train_labels = train_df['labels'].values

test_texts = test_df['text'].values
test_labels = test_df['labels'].values
```

Compared to simpletransformers, we get a closer look at what happens
‘under the hood’ with huggingface. We will see the transformation of the
text better — each tweet will be truncated if they’re more than 512
tokens or padded if they’re fewer than 512 tokens.

The tokens will be separated into “word pieces” using the transformers
tokenizers (‘DistilBertTokenizerFast’ in this case to match the
DistiBERT model). And some special tokens will also be added such as
**CLS** (start token of every tweet) and **SEP** (separator between each
sentence {not tweet}):

``` python
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
```

We now encode our texts using the tokenizer.

``` python
from datasets import Dataset

train_df = Dataset.from_pandas(train_df)
test_df = Dataset.from_pandas(test_df)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_train_df = train_df.map(tokenize_function, batched=True)
tokenized_test_df = test_df.map(tokenize_function, batched=True)
```

    Map:   0%|          | 0/160 [00:00<?, ? examples/s]

    Map:   0%|          | 0/40 [00:00<?, ? examples/s]

We now load the DistilBERT model and specify that it should use the GPU.

``` python
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_)).to()
```

As we did with simpletransformers, we now set the training parameters,
i.e., the number of epochs.

``` python
import accelerate

training_args = TrainingArguments(
    num_train_epochs=3,              # total number of training epochs
    output_dir='./results',          # output directory
    report_to='none'
)
```

#### Fine-tune the DistilBERT model

First, we define a custom evaluation function that returns the accuracy.
You could modify this function to return precision, recall, F1, and/or
other metrics.

``` python
from sklearn.metrics import accuracy_score
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
  }
```

Then we create a HuggingFace `Trainer` object using the
`TrainingArguments` object that we created above. We also send our
`compute_metrics` function to the `Trainer` object, along with our test
and train datasets.

``` python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_df,
    compute_metrics=compute_metrics
)
```

Line 2  
the instantiated 🤗 Transformers model to be trained

Line 3  
training arguments, defined above

Line 4  
training dataset

Line 5  
our custom evaluation function

Time to finally fine-tune!

``` python
trainer.train()
```

    <div>
      &#10;      <progress value='60' max='60' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [60/60 02:58, Epoch 3/3]
    </div>
    &#10;

| Step | Training Loss |
|------|---------------|

<p>

    TrainOutput(global_step=60, training_loss=0.39763174057006834, metrics={'train_runtime': 181.3958, 'train_samples_per_second': 2.646, 'train_steps_per_second': 0.331, 'total_flos': 63584351354880.0, 'train_loss': 0.39763174057006834, 'epoch': 3.0})

#### Save fine-tuned model

The following cell will save the model and its configuration files to a
directory in Colab. To preserve this model for future use, you should
download the model to your computer.

``` python
trainer.save_model(cached_model_directory_name)
```

(Optional) If you’ve already fine-tuned and saved the model, you can
reload it using the following line. You don’t have to run fine-tuning
every time you want to evaluate.

``` python
# trainer = DistilBertForSequenceClassification.from_pretrained(cached_model_directory_name)
```

We can now evaluate the model by predicting the labels for the test set.

``` python
predicted_results = trainer.predict(tokenized_test_df)
```

``` python
predicted_labels = predicted_results.predictions.argmax(-1) # Get the highest probability prediction
predicted_labels = predicted_labels.flatten().tolist()      # Flatten the predictions into a 1D list
predicted_labels[0:5]
```

    [1, 1, 1, 1, 1]

``` python
print(classification_report(tokenized_test_df['labels'],
                            predicted_labels))
```

                  precision    recall  f1-score   support

               0       0.94      0.75      0.83        20
               1       0.79      0.95      0.86        20

        accuracy                           0.85        40
       macro avg       0.86      0.85      0.85        40
    weighted avg       0.86      0.85      0.85        40

You can now use this classifier on other types of data to label it for
potentially sexist content.

## Optional: Multi-class classification

In the previous parts, we finetuned a binary classifier for
differentiating sexist vs. non-sexist content. However, the CMSB dataset
has fine-grained labels for sexism based on **content** and
**phrasing**.

So we now use a multi-class classifier using simpletransformers, with a
few tweaks to our earlier code.

But first, we have to aggregate the annotations from all crowdworkers to
obtain the content and phrasing labels. For simplicity, we will use the
majority label (breaking ties randomly).

``` python
sexism_data_annotations = pd.read_csv('sexism_data/sexism_annotations.csv', sep = ',')
sexism_data_annotations.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | phrasing | content | worker | id   |
|-----|----------|---------|--------|------|
| 0   | 3        | 2       | 0      | 1815 |
| 1   | 3        | 6       | 1      | 1815 |
| 2   | 3        | 6       | 2      | 1815 |
| 3   | 3        | 6       | 3      | 1815 |
| 4   | 3        | 6       | 4      | 1815 |

</div>

``` python
tweets = sexism_data_annotations['id'].unique()
```

``` python
from collections import Counter

content_labels = []
phrasing_labels = []

for tweet in tweets:
    data_subset = sexism_data_annotations[sexism_data_annotations['id'] == tweet]
    content_labels.append(Counter(data_subset['content'].values).most_common()[0][0])
    phrasing_labels.append(Counter(data_subset['phrasing']).most_common()[0][0])
```

Line 8  
get the majority label for content

Line 9  
get the majority label for phrasing

``` python
finegrained_sexism_data = pd.DataFrame([tweets, content_labels, phrasing_labels]).T
finegrained_sexism_data.columns = ['id', 'content_label', 'phrasing_label']
finegrained_sexism_data
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|      | id    | content_label | phrasing_label |
|------|-------|---------------|----------------|
| 0    | 1815  | 6             | 3              |
| 1    | 8199  | 2             | 3              |
| 2    | 11847 | 6             | 3              |
| 3    | 9218  | 6             | 3              |
| 4    | 13298 | 6             | 3              |
| ...  | ...   | ...           | ...            |
| 5645 | 2383  | 6             | 2              |
| 5646 | 5627  | 6             | 3              |
| 5647 | 11041 | 6             | 3              |
| 5648 | 3535  | 6             | 3              |
| 5649 | 9901  | 6             | 3              |

<p>5650 rows × 3 columns</p>
</div>

``` python
finegrained_sexism_data.groupby('content_label').size()
```

    content_label
    1     625
    2     876
    3     173
    4      78
    5     237
    6    3661
    dtype: int64

``` python
finegrained_sexism_data.groupby('phrasing_label').size()
```

    phrasing_label
    1     149
    2     223
    3    5278
    dtype: int64

The six content and three phrasing categories are:

![](img/img1.png)

Let’s join this data with the tweets data from ‘all_data.csv’

``` python
finegrained_sexism_data = pd.merge(finegrained_sexism_data, sexism_data[['id', 'text', 'sexist']])
```

``` python
finegrained_sexism_data.groupby(['content_label']).size()
```

    content_label
    1    37
    2    53
    3     6
    4     1
    5     1
    6    39
    dtype: int64

Since our dataset is somewhat imbalanced with low representation for
some categories, we can restrict it to only those classes that have at
least 30 instances, i.e., 1, 2, and 6.

``` python
finegrained_sexism_data = finegrained_sexism_data[finegrained_sexism_data['content_label'].isin([1, 2, 6])]

# we also change the label range for simpletransformers, making them range from 0 to 2.
label_map = {1 : 0,
             2 : 1,
             6 : 2}
finegrained_sexism_data['content_label'] = [label_map[i] for i in finegrained_sexism_data['content_label']]
finegrained_sexism_data.groupby(['content_label']).size()
```

    content_label
    0    37
    1    53
    2    39
    dtype: int64

Let’s train a classifier for identifying sexist content or phrasing

``` python
category = 'content_label'
```

``` python
multi_train_df, multi_test_df = train_test_split(finegrained_sexism_data,
                                                 stratify=finegrained_sexism_data[category],
                                                 test_size=0.2)
```

You have the add the number of labels to the model initialization.

``` python
# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=5,
                                output_dir='output_st',
                                overwrite_output_dir=True)

# Create a ClassificationModel
model = ClassificationModel(
    "distilbert", "distilbert-base-uncased", num_labels=len(finegrained_sexism_data[category].unique()),
    use_cuda=gpu_avail,
    args=model_args
)


# we set some additional parameters when using a GPU
if gpu_avail:
    model_args.use_multiprocessing=False
    model_args.use_multiprocessing_for_evaluation=False
```

``` python
# multi_train_df['content_label'] = [i-1 for i in multi_train_df['content_label']]
# multi_test_df['content_label'] = [i-1 for i in multi_test_df['content_label']]
```

``` python
multi_train_df = multi_train_df[['text', category]]
multi_test_df = multi_test_df[['text', category]]
```

``` python
# Train the model.
model.train_model(multi_train_df)
```

    Epoch:   0%|          | 0/5 [00:00<?, ?it/s]

    Running Epoch 1 of 5:   0%|          | 0/13 [00:00<?, ?it/s]

    Running Epoch 2 of 5:   0%|          | 0/13 [00:00<?, ?it/s]

    Running Epoch 3 of 5:   0%|          | 0/13 [00:00<?, ?it/s]

    Running Epoch 4 of 5:   0%|          | 0/13 [00:00<?, ?it/s]

    Running Epoch 5 of 5:   0%|          | 0/13 [00:00<?, ?it/s]

    (65, 0.8820166963797349)

``` python
predictions, raw_outputs = model.predict([sexist_tweet])
predictions
```

      0%|          | 0/1 [00:00<?, ?it/s]

    array([1])

``` python
preds, _ = model.predict(list(multi_test_df['text'].values))
```

      0%|          | 0/1 [00:00<?, ?it/s]

``` python
print(classification_report(multi_test_df[category], preds))
```

                  precision    recall  f1-score   support

               0       0.67      0.29      0.40         7
               1       0.60      0.82      0.69        11
               2       0.75      0.75      0.75         8

        accuracy                           0.65        26
       macro avg       0.67      0.62      0.61        26
    weighted avg       0.66      0.65      0.63        26

We can see that the model performs worse than binary sexism
classification, but still better than a random chance model which would
have add an accuracy of 0.3 as we have three classes.

<div id="refs" class="references csl-bib-body hanging-indent"
entry-spacing="0" line-spacing="2">

<div id="ref-samory2021call" class="csl-entry">

Samory, M., Sen, I., Kohne, J., Flöck, F., & Wagner, C. (2021). “Call me
sexist, but...”: Revisiting sexism detection using psychological scales
and adversarial samples. *<span class="nocase">Proceedings of the
international AAAI conference on web and social media</span>*, *15*,
573–584. <https://doi.org/10.1609/icwsm.v15i1.18085>

</div>

<div id="ref-sanh2019distilbert" class="csl-entry">

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a
distilled version of BERT: Smaller, faster, cheaper and lighter. *arXiv
Preprint arXiv:1910.01108*. <https://arxiv.org/abs/1910.01108>

</div>

</div>
