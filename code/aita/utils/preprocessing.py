import string
import re
import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from torchtext.vocab import vocab

URL_REGEX = '(http|https)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]'
NON_ALPHA_NUMERIC_REGEX = '[^a-zA-Z0-9- ]'
TAG_REGEX = '@[^\\s]*'
NUM_CLASSES = 2
AHOLE_CLASSES = ['yta', 'nta']
MAX_SEQ_LENGTH = 128

def load_dataset(filepath: str, classes: list, tokenizer, rm_punct: bool = False):
    """
    Load dataset for BERT, and applies corresponding pre-processing.
    :param filepath:
    :param classes: of labels, yta and nta
    :param tokenizer: BERT tokenizer used
    :param rm_punct: to either remove or not punctuation
    :return: ids, and attention vectors, and labels
    """
    label2index = {x: i for i, x in enumerate(classes)}

    texts, labels = [], []
    for class_label in classes:
        df = pd.read_csv(filepath.format(class_label), usecols=['body'])
        for text in df['body']:
            # required for empty descriptions in the Tumblr dataset
            try:
                tokens = process_text(text, tokenizer, rm_punct)
                if len(tokens):
                    texts.append(tokens)
                    labels.append(label2index[class_label])
            except TypeError:
                continue

    return texts, labels


def process_text(text, tokenizer, rm_punct: bool = False):
    """
    Pre-processing applyied on teh text: removes URLs, tags, and web links, or labels appearing
    in the text.
    :param text:
    :param tokenizer: BERT
    :param rm_punct: remove punctuation
    :return: corresponding tokens for the proposed text
    """
    if type(text) is not str:
        raise TypeError('Text is not of type string')

    # remove special string from the text: URLs and emojis (by encoding and decoding to/from ascii)
    text = remove_urls(text)
    text = remove_tags(text)

    # remove unknown characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    # remove all punctuation
    if rm_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = tokenizer.tokenize(text)
    # remove all labels from texts
    tokens = delete_label_word(tokens, 'YTA')
    tokens = delete_label_word(tokens, 'NTA')

    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    return tokens

def process_text_without_tokenise(text, rm_punct: bool = False):
    """
    Pre-process text for models different from BERT, like word2vec or LSTM
    :param text:
    :param rm_punct:
    :return: cleaned version of text
    """
    if type(text) is not str:
        raise TypeError('Text is not of type string')

    # remove special string from the text: URLs and emojis (by encoding and decoding to/from ascii)
    text = remove_urls(text)
    text = remove_tags(text)
    text = remove_non_alphanumeric(text)
    text = text.strip()
    # remove text white space on the sides
    text = " ".join(re.split("\s+", text, flags=re.UNICODE))


    # remove unknown characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    if rm_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    # remove all punctuation
    return text

# next functions are self-explanatory
def to_indices_vector(post, vocab, max_length=256):
    encoded = np.zeros(max_length, dtype=int)
    indices = vocab(post)
    length = min(max_length, len(indices))
    encoded[:length] = indices[:length]
    return encoded, length

def get_vocab(posts):
    counter_list = []
    for body in posts["body"]:
        for token in body:
            counter_list.append(token)

    counter = Counter(counter_list)
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    v1 = vocab(ordered_dict, specials=['<unk>'])
    v1.set_default_index(v1['<unk>'])

    return v1

def remove_non_alphanumeric(text: str) -> str:
    return re.sub(NON_ALPHA_NUMERIC_REGEX, ' ', text)


def remove_urls(text: str) -> str:
    return re.sub(URL_REGEX, ' ', text)


def remove_tags(text: str) -> str:
    return re.sub(TAG_REGEX, ' ', text)


def delete_label_word(words: list, label: str) -> list:
    while label in words:
        words.remove(label)
    return words