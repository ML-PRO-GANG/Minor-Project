import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import os
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification
from scipy.special import softmax
import shutil


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

plt.style.use('ggplot')

df = pd.read_csv('Womens Clothing E-Commerce Reviews .csv.zip')
print(df.shape)
df = df.head(500)
print(df.shape)

df.head()

ax = df['Rating'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()

example = df['Review Text'][50]
print(example)

nltk.download('punkt')

#my terminal asked to dwnld
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')

tokens = nltk.word_tokenize(example)
tokens[:10]

nltk.download('averaged_perceptron_tagger')

tagged = nltk.pos_tag(tokens)
tagged[:10]

nltk.download('maxent_ne_chunker')
nltk.download('words')
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

sia.polarity_scores('I am so happy!')
sia.polarity_scores('This is the worst thing ever.')
sia.polarity_scores(example)

# Assuming 'data' is your DataFrame
# Handle missing or NaN values in the entire column at once
df['Review Text'] = df['Review Text'].fillna('')

# Ensure all reviews are strings in the entire column at once
df['Review Text'] = df['Review Text'].astype(str)

# Store the results in a dictionary or directly into a new column
res = {}

for index, row in df.iterrows():
    text = row['Review Text']  # Now text will always be a string
    myid = row['Clothing ID']
    res[myid] = sia.polarity_scores(text)

# If you want to add the sentiment scores to the DataFrame
df['sentiment_scores'] = df['Review Text'].apply(lambda x: sia.polarity_scores(x))

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Clothing Id'})
# Merge based on 'Clothing Id' in vaders and 'Clothing ID' in df
vaders = vaders.merge(df, how='left', left_on='Clothing Id', right_on='Clothing ID')

vaders.head()

ax = sns.barplot(data=vaders, x='Rating', y='compound')
ax.set_title('Compund Score by Amazon Star Review')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Rating', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Rating', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Rating', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

print(vaders.columns)
print(df.columns)
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"

# Delete cache files from the default cache directory
cache_dir = os.path.join(os.getenv('HOME', ''), '.cache', 'huggingface', 'transformers')
shutil.rmtree(cache_dir, ignore_errors=True)

# Retry loading the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# VADER results on example
print(example)
sia.polarity_scores(example)

# Run for Roberta Model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Review Text']
        myid = row['Clothing ID']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')

results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Clothing ID'})
results_df = results_df.merge(df, how='left')

results_df.columns

sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Rating',
            palette='tab10')
plt.show()

results_df.query('Rating == 1') \
    .sort_values('roberta_pos', ascending=False)['Review Text'].values[0]

results_df.query('Rating == 1') \
    .sort_values('vader_pos', ascending=False)['Review Text'].values[0]

# nevative sentiment 5-Star view
results_df.query('Rating == 5') \
    .sort_values('roberta_neg', ascending=False)['Review Text'].values[0]
results_df.query('Rating == 5') \
    .sort_values('vader_neg', ascending=False)['Review Text'].values[0]

results_df.query('Rating == 5') \
    .sort_values('vader_neg', ascending=False)['Review Text'].values[0]

sent_pipeline = pipeline("sentiment-analysis")
sent_pipeline('I love sentiment analysis!')
sent_pipeline('i love this dress')