import json

import praw
import os
import argparse
import pickle
import torch
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
from sklearn.naive_bayes import GaussianNB

from aita.models import BERTJoint
from aita.utils.preprocessing import process_text
from transformers import BertForSequenceClassification, BertTokenizer

load_dotenv(dotenv_path=Path('dev.env'))

APP_ID = os.getenv('APP_ID')
APP_SECRET = os.getenv('APP_SECRET')
AGENT = os.getenv('AGENT')
USERNAME = os.getenv('USERNAME')
PASSWORD = os.getenv('PASSWORD')

reddit = praw.Reddit(
    client_id=APP_ID,
    client_secret=APP_SECRET,
    user_agent=AGENT,
)

config_file = open('infer_config.json')
config = json.load(config_file)

def process_comments(sub):

    comments_list = []
    for comment in sub.comments:
        discard = "bot" in comment.body or len(comment.body.split()) < 3
        if not discard:
            comments_list.append(comment.body)

    return comments_list[:3]


def run_inference(post_url, include_comments):
    print(f"Getting The post from {post_url} \nInclude Comments: {include_comments}")
    sub = reddit.submission(url=post_url)
    sub.comment_limit = 5
    sub.comment_sort = 'top'
    sub.comments.replace_more(limit=0)
    print(f"Obtained Post: {sub.title}")

    if not include_comments:
        predict_no_comment(sub)
    else:
        predict_with_comments(sub)

def predict_with_comments(sub):
    text_body = sub.selftext
    text_comments = process_comments(sub)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    tokenized_text = process_text(text_body, tokenizer, True)
    encoded_dict = tokenizer.encode_plus(tokenized_text,
                                         add_special_tokens=False,
                                         truncation=True,
                                         max_length=512,
                                         padding='max_length',
                                         return_attention_mask=True,
                                         return_tensors='pt')

    text_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    comments_ids, comments_attention = [], []
    for comment in text_comments:
        encoded_dict = tokenizer.encode_plus(comment,
                                             add_special_tokens=False,
                                             truncation=True,
                                             max_length=128,
                                             pad_to_max_length=True,
                                             return_attention_mask=True,
                                             return_tensors='pt')

        comments_ids.append(encoded_dict['input_ids'])
        comments_attention.append(encoded_dict['attention_mask'])

    comments_ids = torch.cat(comments_ids, dim=0)
    comments_attention = torch.cat(comments_attention, dim=0)
    model = BERTJoint(posts_weights=config["post_weights"], comments_weights=config["comments_weights"], device="cpu")
    mlp_file = open(config["gaussian_model_path"], "rb")
    mlp = pickle.load(mlp_file)

    embeddings = model(post_id=text_ids,
                       post_attention=attention_mask,
                       comments_ids=comments_ids,
                       comments_attentions=comments_attention)

    y_pred = mlp.predict(np.expand_dims(embeddings, 0))
    print("Prediction value", y_pred)
    if y_pred == 1:
        print("You are the Asshole")
    else:
        print("You are not the Asshole")

def predict_no_comment(sub):
    text_body = sub.selftext
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    model.load_state_dict(torch.load(config["posts_weights"]))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_text = process_text(text_body, tokenizer)
    encoded_dict = tokenizer.encode_plus(tokenized_text,
                                         add_special_tokens=False,
                                         truncation=True,
                                         max_length=512,
                                         padding='max_length',
                                         return_attention_mask=True,
                                         return_tensors='pt')
    text_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    with torch.no_grad():
        prediction = model(text_ids,
                           token_type_ids=None,
                           attention_mask=attention_mask,
                           labels=None,
                           return_dict=False)

        predicted_label_id = prediction[0].argmax().to('cpu').numpy()
        if predicted_label_id == 1:
            print("You are the Asshole")
        else:
            print("You are not the Asshole")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict AITA for a post')
    parser.add_argument("-u", "--url", type=str, dest="url")
    parser.add_argument("-c", "--include-comments", action="store_true", dest="include_comments")
    args = parser.parse_args()
    print("AITA Predictor \n\n")
    run_inference(args.url, args.include_comments)
