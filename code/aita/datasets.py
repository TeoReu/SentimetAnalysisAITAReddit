import pandas as pd
from torch.utils.data import Dataset
import torch
from aita.utils.preprocessing import process_text
import numpy as np


class AITADataset(Dataset):
    """
    Load data for the LSTM module takes numpy arrays of embedding tokens and the labels
    of the corresponding posts, and prepares them in order to be fitted by a benchmark classfier
    such as Naive Bayes or Multi Layer Perceptron
    """
    def __init__(self, X, Y) -> None:
        super().__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.X[idx][1], torch.tensor([self.Y[idx]],
                                                                                               dtype=torch.float32)

def perform_undersample(df, column, label_1, label_2):
    """
    Returns a balanced dataset in which for each label we have approx. equal number
    of posts
    :param df: th input dataframe
    :param column: the name of the column where the posts, or comment lies
    :param label_1: number of label 1 posts
    :param label_2: number of label 2 posts
    :return:
    """
    df_c1 = df[df[column] == label_1]
    df_c2 = df[df[column] == label_2]
    n = df_c1.shape[0]
    df_c2 = df_c2.iloc[:n]
    df = pd.concat([df_c1, df_c2], axis=0)
    
    return df.sample(frac=1).reset_index(drop=True)


class AITADatasetBERT(Dataset):
    """
    Loader of the dataset for a simple BERT model. Taxes the dataset path, and corresponding tokenizer
    and return a dictionary with all variables needed in order to perform training.
    """
    def __init__(self, dataset_path, tokenizer, max_token_length, undersample=False):
        super().__init__()
        self.dataset = pd.read_csv(dataset_path)

        self.dataset['body'] = self.dataset['body'].astype(str)
        self.dataset['is_yta'] = self.dataset['is_yta'] == "yta"
        
        if undersample:
            self.dataset = perform_undersample(self.dataset, "is_yta", True, False)
    

        self.dataset['is_yta'] = self.dataset['is_yta'].apply(lambda label: [label, not label])

        self.texts = self.dataset['body'].to_list()
        self.labels = self.dataset['is_yta'].to_list()

        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text = self.texts[item]
        text = process_text(text=text, tokenizer=self.tokenizer, rm_punct=True)
        encoded_dict = self.tokenizer.encode_plus(text,
                                                  add_special_tokens=False,
                                                  truncation=True,
                                                  max_length=self.max_token_length,
                                                  padding='max_length',
                                                  return_attention_mask=True,
                                                  return_tensors='pt')
        labels = self.labels[item]
        labels = torch.tensor(labels)
        labels = labels.bool().int().float()

        return encoded_dict['input_ids'].squeeze(), encoded_dict['attention_mask'].squeeze(), labels


class AITADatasetJoint(Dataset):
    """
    Loader of the dataset for a joint BERT model. Taxes the dataset path, and corresponding tokenizer
    and return a dictionary with all variables needed in order to perform training, including
    comments tokens ids and attention vectors
    """
    def __init__(self, dataset_path, tokenizer, max_token_length_posts=512, max_token_length_comments=128, undersample=False):
        super().__init__()
        self.dataset = pd.read_csv(dataset_path)

        self.dataset['body'] = self.dataset['body'].astype(str)
        if undersample:
            self.dataset = perform_undersample(self.dataset, "is_yta", "yta", "nta")
        

        self.dataset['is_yta'] = self.dataset['is_yta'] == "yta"
        self.dataset['is_yta'] = self.dataset['is_yta'].apply(lambda label: [label, not label])
        self.labels = self.dataset['is_yta'].to_list()

        self.dataset['c1'] = self.dataset['c1'].astype(str)
        self.dataset['c2'] = self.dataset['c2'].astype(str)
        self.dataset['c3'] = self.dataset['c3'].astype(str)
        self.posts = self.dataset['body'].to_list()

        self.comments_list = []
        for _, row in self.dataset.iterrows():
            self.comments_list.append([row["c1"], row["c2"], row["c3"]])
        

        self.tokenizer = tokenizer
        self.max_token_length_posts = max_token_length_posts
        self.max_token_length_comments = max_token_length_comments

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        post = self.posts[item]
        post = process_text(text=post, tokenizer=self.tokenizer, rm_punct=True)
        post_encoded_dict = self.tokenizer.encode_plus(post,
                                                  add_special_tokens=False,
                                                  truncation=True,
                                                  max_length=self.max_token_length_posts,
                                                  pad_to_max_length=True,
                                                  return_attention_mask=True,
                                                  return_tensors='pt')
        comments = self.comments_list[item]
        comments_ids, comments_attention = [], []
        for comment in comments:
            encoded_dict = self.tokenizer.encode_plus(comment,
                                                      add_special_tokens=False,
                                                      truncation=True,
                                                      max_length=self.max_token_length_comments,
                                                      pad_to_max_length=True,
                                                      return_attention_mask=True,
                                                      return_tensors='pt')

            comments_ids.append(encoded_dict['input_ids'])
            comments_attention.append(encoded_dict['attention_mask'])

        comments_ids = torch.cat(comments_ids, dim=0)
        comments_attention = torch.cat(comments_attention, dim=0)

        labels = self.labels[item]
        labels = torch.tensor(labels)
        labels = labels.bool().int().float()

        return (post_encoded_dict['input_ids'].squeeze(),
                post_encoded_dict['attention_mask'].squeeze(),
                comments_ids,
                comments_attention,
                labels)
