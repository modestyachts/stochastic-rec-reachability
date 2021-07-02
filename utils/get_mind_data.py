from io import BytesIO
import json
import os
import pandas as pd
from urllib.request import urlopen
from zipfile import ZipFile

from config import DATAPATH

# Make directories for mind data
mind_dir = os.path.join(DATAPATH, 'mind')
if not os.path.exists(mind_dir):
    os.makedirs(mind_dir)
if not os.path.exists(os.path.join(mind_dir, 'large')):
    os.makedirs(os.path.join(mind_dir, 'large'))
if not os.path.exists(os.path.join(mind_dir, 'small')):
    os.makedirs(os.path.join(mind_dir, 'small'))


# Download MIND data:
# this extracts in each folder 4 files: behaviours.tsv, news.tsv, entity_embedding.vec and relation_embedding.vec
# for more detailed dataset description see https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md
# data_types = [('large', 'train'), ('large', 'test'), ('large', 'dev'), ('small', 'train'), ('small', 'dev')]
# for size, kind in data_types:
#     zipurl = 'https://mind201910small.blob.core.windows.net/release/MIND{}_{}.zip'.format(size, kind)
#     dest_folder = os.path.join(mind_dir, size, kind)
#     if not os.path.exists(dest_folder):
#         os.makedirs(dest_folder)
#     with urlopen(zipurl) as zipresp:
#         with ZipFile(BytesIO(zipresp.read())) as zfile:
#             zfile.extractall(dest_folder)

# Preprocess data
def process_entities(row, kind = 'Title_Entities'):
    entities = row[kind]
    if pd.isnull(entities):
        return []
    entities = json.loads(entities)
    return entities

def get_wiki_list(row):
    entities = row['Total_Entities']
    wiki_list = [entity['WikidataId'] for entity in entities]
    return list(set(wiki_list))

def get_len(row, how = 'Impressions'):
    row_str = row[how]
    if row_str == '':
        return 0
    if isinstance(row_str, float):
        return row_str
    row_arr = row_str.split(' ')
    return len(row_arr)

def get_clicks(row):
    row_str = row['Impressions']
    row_arr = row_str.split(' ')
    clicks = [arr for arr in row_arr if arr[-1] == '1']
    return(len(clicks))

def get_seen_clicks_history_news(row):
    row_str = row['Impressions']
    row_arr = row_str.split(' ')
    history_str = row['History']
    if isinstance(history_str, float):
        history = []
    else:
        history = history_str.split(' ')
    seen =[arr[:-2] for arr in row_arr]
    clicks = [arr[:-2] for arr in row_arr if arr[-1] == '1']
    non_clicks = [arr[:-2] for arr in row_arr if arr[-1] == '1']
    return seen, clicks, non_clicks, history

def flatten(t, remove_duplicates = True):
    res = [item for sublist in t for item in sublist]
    if remove_duplicates:
        return list(set(res))
    else:
        return res

def combine(df, col='Seen_Items'):
    all_items = flatten(df[col])
    return all_items

def make_df(mind_dict, rating = 1):
    users = flatten([[uid]*len(iids) for uid, iids in mind_dict.items()], remove_duplicates=False)
    items = flatten([iids for uid, iids in mind_dict.items()], remove_duplicates=False)
    df = pd.DataFrame({})
    df['user_id'] = users
    df['item_id'] = items
    df['rating'] = rating
    return df

for size, kind in data_types:
    # Process news data
    news_filename = os.path.join(mind_dir, size, kind, "news.tsv")
    news_col_names = ['item_id', "Category", "Subcategory", "Title", "Abstract",
                      "URL", "Title_Entities", "Abstract_Entities"]
    news_df = pd.read_csv(news_filename, sep = "\t", names = news_col_names, header=None)

    news_df['Title_Entities'] = news_df.apply(process_entities, axis = 1)
    news_df['Abstract_Entities'] = news_df.apply(process_entities, axis = 1, kind = 'Abstract_Entities')
    news_df['Total_Entities'] = news_df['Title_Entities'] + news_df['Abstract_Entities']
    news_df['Wiki_List'] = news_df.apply(get_wiki_list, axis = 1)

    keep_cols = ['item_id', "Category", "Subcategory", "Title_Entities", "Abstract_Entities", "Wiki_List"]
    processed_news_df = news_df[keep_cols]

    processed_news_filename = os.path.join(mind_dir, size, kind, 'processed_news.tsv')
    processed_news_df.to_csv(processed_news_filename, sep = '\t', index = False, quotechar='"')

    # Process behaviour data
    behaviours_filename = os.path.join(mind_dir, size, kind, 'behaviors.tsv')
    behaviour_col_names = ['ImpressionID', 'UID', 'Time', 'History', 'Impressions']
    behaviour_df = pd.read_csv(behaviours_filename, sep = '\t', names = behaviour_col_names, header=None)

    behaviour_df['Num_Impressions'] = behaviour_df.apply(get_len, axis = 1)
    behaviour_df['History_Len'] = behaviour_df.apply(get_len, axis = 1, how ='History')
    behaviour_df['Num_Clicks'] = behaviour_df.apply(get_clicks, axis = 1)
    behaviour_df[['Seen_Items', 'Clicked_Items', 'Nonclicked_Items', 'History_Items']] = behaviour_df.apply(get_seen_clicks_history_news,
                                                                   axis =1, result_type = 'expand')
    # groupby user_id and concatenate the seen items and clicked items
    user_groups = behaviour_df.groupby('UID')
    clicked_dict = dict(user_groups.apply(combine, col='Clicked_Items'))
    history_dict = dict(user_groups.apply(combine, col='History_Items'))
    non_clicked_dict = dict(user_groups.apply(combine, col='Nonclicked_Items'))

    clicked_df = make_df(clicked_dict, rating = 1)
    non_clicked_df = make_df(non_clicked_dict, rating = 0)
    history_df = make_df(history_dict, rating = 1)

    clicked_df_filename = os.path.join(mind_dir, size, kind, 'clicked.csv')
    non_clicked_df_filename = os.path.join(mind_dir, size, kind,'non_clicked.csv')
    history_df_filename = os.path.join(mind_dir, size, kind, 'history.csv')

    clicked_df.to_csv(clicked_df_filename, index=False)
    non_clicked_df.to_csv(non_clicked_df_filename, index=False)
    history_df.to_csv(history_df_filename, index=False)
