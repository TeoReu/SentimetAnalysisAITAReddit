"""
collect posts with pushshif.io API
"""
from psaw import PushshiftAPI
import praw
import csv
import datetime as ds
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path('dev.env'))
after = int(ds.datetime(2019, 1, 1).timestamp())
before = int(ds.datetime(2020, 1, 1).timestamp())

APP_ID = os.getenv('APP_ID')
APP_SECRET = os.getenv('APP_SECRET')
AGENT = os.getenv('AGENT')
# Initialize PushShift
reddit = praw.Reddit(
    client_id=APP_ID,
    client_secret=APP_SECRET,
    user_agent=AGENT
)
api = PushshiftAPI(reddit)
submission_psaw = api.search_submissions(subreddit='AmITheAsshole',
                                         after=after,
                                         before=before,
                                         filter=['id', 'title', 'selftext', 'link_flair_text'])

fields = ["id", "title", "body", "is_yta"]
csv_file = open('data/posts.csv', 'w')
writer = csv.DictWriter(csv_file, fieldnames=fields)
writer.writeheader()
count = 0
for submission in submission_psaw:
    yta = False
    discard = submission.selftext in ["[removed]", "[deleted]"] or \
              not submission.link_flair_text or \
              submission.link_flair_text not in ["Not the A-hole", "Asshole"]
    if not discard:
        is_yta = "yta" if submission.link_flair_text == "Asshole" else "nta"
        values = {"id": submission.id, "title": submission.title, "body": submission.selftext,
                  "is_yta": is_yta}
        writer.writerow(values)
        count += 1
        if count % 100 == 0:
            print(f"Current Count {count}")
