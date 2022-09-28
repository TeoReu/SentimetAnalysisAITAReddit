import pandas as pd
import numpy as np
import matplotlib
from collections import Counter

'''
matplotlib.use("pgf")

matplotlib.rcParams.update({
"pgf.texsystem": "pdflatex",
'font.family': 'serif',
'text.usetex': True,
'pgf.rcfonts': False,
})'''
import matplotlib.pyplot as plt
import nltk
import re
import string
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

# %%
# Test
# Results for these scripts can be viewed in statistics/

merged_dataset = pd.read_csv("data/merged_mod.csv")
x = np.ones((1, 1))


# %%
def histograms_length(type):
    """
    Histograms of the length of posts/comments
    :param type: comments or posts
    :return:
    """
    length = merged_dataset[type].apply(lambda a: len(str(a).split()))
    plt.hist(length, 100, density=True, facecolor='pink', alpha=0.75)
    if type == 'body':
        plt.xlim(40, 1000)
    else:
        plt.xlim(10, 300)
    plt.xlabel('Number of words')
    plt.ylabel('Ratio')

    if type == 'body':
        type = 'posts'
    plt.title('Histogram of length of ' + type)
    plt.savefig('latex_plots/histogram_' + type + '.png')
    plt.show()


histograms_length('body')
histograms_length('comments')

# %%
# Most popular words for each label type

stop_words = stopwords.words('english')


def cleaning(text):
    """
    Cleaning text in order to get most popular words, also sto
    words removal
    :param text: test of posts
    :return: cleaned text
    """
    # converting to lowercase, removing URL links, special characters, punctuations...
    text = text.lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('[“”…]', '', text)

    # removing the emojies               # https://www.kaggle.com/alankritamishra/covid-19-tweet-sentiment-analysis#Sentiment-analysis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # removing the stop-words
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stop_words]
    filtered_sentence = (" ").join(tokens_without_sw)
    text = filtered_sentence

    return text


def most_popular_words(yta):
    """
    Most popular words for each label type
    :param yta: label ype: whether yta or nta
    :return: saves cvs with most popular words for each label type
    """
    md = merged_dataset.dropna()
    new_df = md.mask(md['is_yta'] ==yta).dropna()
    dt = new_df['body'].apply(cleaning)
    p = Counter(" ".join(dt).split()).most_common(50)
    rslt = pd.DataFrame(p, columns=['Word', 'Frequency'])
    rslt.to_csv('statistics/most_common_words_'+yta+'_posts.csv')


most_popular_words('yta')
most_popular_words('nta')

# %%
# Ratio of yta, nta labels on the posts
def ratio_label():
    """
    Saves a df with the ratio of each label type over the dataset
    :return:
    """
    md = merged_dataset.dropna()
    df_nta = md.mask(md['is_yta'] == 'yta').dropna()
    df_yta = md.mask(md['is_yta'] == 'nta').dropna()

    data = [['Count', len(df_yta), len(df_nta)], ['Ratio', len(df_yta)/len(md), len(df_nta)/len(md)]]

    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=['', 'YTA', 'NTA'])

    df.to_csv('statistics/yta_ratio_merged_dataset.csv')

ratio_label()

# %%
# Agreement between the comments and the post's label
def agreement():
    """
    Saves df with agreement levels between comments and posts levels
    :return:
    """
    md = merged_dataset.dropna()
    df_nta = md.mask(md['is_yta'] == 'yta').dropna()
    df_yta = md.mask(md['is_yta'] == 'nta').dropna()

    agreement_on_nta = df_nta['agreement'].value_counts()/len(df_nta)
    agreement_on_yta = df_yta['agreement'].value_counts()/len(df_yta)

    result = pd.concat([agreement_on_nta, agreement_on_yta], axis=1)
    result.columns = ['nta_agreement', 'yta_agreement']

    result.to_csv('statistics/agreement_comments_label.csv')