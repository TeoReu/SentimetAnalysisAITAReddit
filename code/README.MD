#### Dev setup

* Create Python3.9 environment by running `python3 -m venv ./nlu-venv`
* Set the interpreter in pycharm (bottom-left corner of IDE)
* Install pip-tools: `python -m pip install pip-tools`
* Generate frozen requirements file: `pip install -r requirements/requirements.txt`
* Iteratively, add newly required package and then run: `pip-compile --upgrade requirements/requirements.in`, and install as above
## Obtain data and models_weights zip at:
https://drive.google.com/drive/folders/1nykCvKZ1qbczKx8TDdUf3JNiQtNn_Ig_?usp=sharing
- unzip files and place the to the project folder replacing the empty folders
## Project Structure Description
### aita: Module which is preoccupied with data pre-processing, datasets building, and models architecture

- datasets.py: Builds AITA dataset
- models.py: Preserves the architecture of our models
- utils/preprocessing.py: Contains functions that help with the pre-processing of the texts


### data: datasets used or previously used for different purposes

- comments files  contain comments from AITA with comment_id, comment_text, and labels
- evaluation files contain data used for evaluation purposes, which the trained models haven't seen before
- merged_mod.py contains a dataframe in which 3 comments, and it's corresponding post are aggregated together
- _nta and_yta files separate dataset based on the labels of the objects


### latex_plots: plots that will latter be used in our report 

Contains histograms of word count distribution over comments and posts

### model_weights: Files with weights for models

We hold weights for following models:
- gaussian_param.pickle: the Naive Bayes classifier used on the joint model that concatenates representations for both comments and posts
- last_model_post.pt: BERT weights for fine-tuned BERT on posts
- LSTMW2V.pth: is for the LSTM model used to get lower representation on posts
- mlp_params.pickle: the Multi Layer Perceptron classifier used on the joint model that concatenates representations for both comments and posts
- weighted_model_posts.pt: BERT that uses Weighted Binary Cross Entropy Loss to get lower space representations

### notebooks: Notebooks used to run our experiments 

-they don't have any presentation purpose, should not be touched

### reddit: Reddit API that gets comments and posts
- uses pushshit.io, to get posts from a time period and extracts corresponding comments based on the post ID
### results: Results obtained after running experiments on each proposed model in part
- some metrics used are: Accuracy, Recall, Precision, F1, Mathew Correlation Coef
- PCA representations
- confusion matrix

### scripts: Used for our presentation

- given a reddit link our code will predict based on the post and top three comments whter post is YTA or not

### statistics: Statistics on the dataset

- analysis of label, length, and approval distributions allover the dataset

### reddit_links
For presentation and demonstration:
Unlabeled
https://www.reddit.com/r/AmItheAsshole/comments/uoqorf/aita_for_giving_my_niece_sleeping_pills_and/
https://www.reddit.com/r/AmItheAsshole/comments/uopplv/aita_for_not_waking_my_husband_up_for_his_5am_walk/
https://www.reddit.com/r/AmItheAsshole/comments/uopjca/aita_for_telling_my_husband_its_time_to_hash/

Labeled
NTA
https://www.reddit.com/r/AmItheAsshole/comments/uo6we1/aita_for_not_giving_up_my_second_free_seat_next/
https://www.reddit.com/r/AmItheAsshole/comments/uo4kes/aita_for_being_harsh_with_my_coworker_while_he/
https://www.reddit.com/r/AmItheAsshole/comments/uoaa4p/aita_telling_my_mom_to_cancel_her_date_to_go_to/
YTA
https://www.reddit.com/r/AmItheAsshole/comments/uo39th/aita_for_calling_my_parents_disgusting_perverts/

### running the script: scripts
From the parent directory:

`Joint BERT Model: scripts/infer_on_post.py -u YOUR_REDDIT_URL -c 
Simple BERT Model: scripts/infer_on_post.py -u YOUR_REDDIT_URL`

