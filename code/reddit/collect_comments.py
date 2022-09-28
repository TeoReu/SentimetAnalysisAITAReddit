"""
collect comments given an id list of posts
"""
import praw
import csv
import datetime as ds
import re
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path('dev.env'))
APP_ID = os.getenv('APP_ID')
APP_SECRET = os.getenv('APP_SECRET')

after = int(ds.datetime(2019, 1, 1).timestamp())
before = int(ds.datetime(2020,1,1).timestamp())
reddit = praw.Reddit(
    client_id = APP_ID,
    client_secret = APP_SECRET,
    user_agent = "zainja_agent"
)
csv_file = open('data/posts.csv', 'r')
reader = csv.DictReader(csv_file)

fields = ["id", "post_id", "body", "is_yta"]
csv_file2 = open('data/c.csv', 'w')
writer = csv.DictWriter(csv_file2, fieldnames=fields)
writer.writeheader()
count = 0
for row in reader:
    submission_praw = reddit.submission(id=row["id"])
    submission_praw.comment_limit = 5
    submission_praw.comment_sort = 'top'
    submission_praw.comments.replace_more(limit=0)

    for comment in submission_praw.comments:
        labelled = "yta" in comment.body.lower() or "nta" in comment.body.lower()
        discard = "bot" in comment.body or not labelled or len(comment.body.split()) < 3
        if not discard:
            is_yta = "yta" if "yta"  in comment.body.lower() else "nta"
            comment.body = re.sub('yta|YTA|NTA|nta', "", comment.body)
            values = {"id": comment.id, "post_id": row['id'], "body": comment.body, "is_yta": is_yta}
            writer.writerow(values)
    count += 1
    if count % 100 == 0:
        print(f'Count {count}')