"""
merges the comments and posts dataframes into a single dataframe
"""
import pandas as pd
#%%
posts = pd.read_csv('data/posts.csv')
comments = pd.read_csv('data/comments.csv')

#%%
comments['len'] = comments['body'].apply(lambda a : len(str(a).split()))
comments = comments[comments['len'] > 3]

comments_count = comments['post_id'].value_counts()
unique_comments = comments_count[comments_count >= 3].index
#%%
filtered = posts[posts['id'].isin(unique_comments)]
filtered_comments = comments[comments['post_id'].isin(filtered['id'])]

yta_len = len(filtered[filtered["is_yta"] == "yta"])
yta_comments_len = len(filtered_comments[filtered_comments["is_yta"] == "yta"])

ratio = yta_len/len(filtered)
commratio = yta_comments_len/len(filtered_comments)
#%%
fl = filtered_comments.groupby(filtered_comments['post_id']).agg(list)
filtered= filtered.rename(columns={"id": "post_id"})
merged = pd.merge(filtered, fl, on="post_id")
merged = merged.rename(columns={"body_x": "body", "is_yta_x": "is_yta", "body_y": "comments", "is_yta_y": "comments_yta",
                                "id": "comment_id"})
merged['comments'] = merged["comments"].apply(lambda a: a[:3])
merged['comments_yta'] = merged["comments_yta"].apply(lambda a: a[:3])
merged["comment_id"] = merged["comment_id"].apply(lambda a: a[:3])

#%%
yta_merged = len(merged[merged["is_yta"] == "yta"])
ratio = yta_merged/len(merged)
#%%
agreements = []
for index, row in merged.iterrows():
    aggreement_count = 0
    for comment in row['comments_yta']:
        if comment == row['is_yta']:
            aggreement_count += 1
    agreements.append(aggreement_count)
        
merged['agreement'] = agreements
c = merged['comments'].to_list()
p_id = merged["post_id"].to_list()

comments = pd.DataFrame(c, columns=["c1", "c2", "c3"])
comments["post_id"] = p_id
merged = pd.merge(merged, comments, on="post_id")
merged = merged.drop(columns=["comments", 'comments_yta'])

merged.to_csv('data/merged_mod.csv')