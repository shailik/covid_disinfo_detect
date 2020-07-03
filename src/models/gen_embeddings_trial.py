'''
This script is a work-in-progress and needs further development.
Right now all of the functions below have successfully run within
a notebook that is being used in tandem to develop this script.
In the next branch, I will continue to build this out, namely by
adding a config file and also by figuring out a more appropriate
means by which to store embeddings, most likely in a database
of some sort (and not within the 'data' sub-folder of the directory)
'''
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from nltk.tokenize import TweetTokenizer
from emoji import demojize
from sentence_transformers import SentenceTransformer
import re
tqdm.pandas()


def path_to_data():
    return Path.cwd() / 'data' / 'dailies'


def normalize_token(token):
    lwrcase_tok = token.lower()
    if token.startswith('@'):
        return '@USER'
    elif lwrcase_tok.startswith('http') or lwrcase_tok.startswith('www'):
        return 'HTTPURL'
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalize_tweet(tweet):
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalize_token(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    normTweet = re.sub(
        r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet
    )
    normTweet = re.sub(
        r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet
    )
    normTweet = re.sub(
        r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet
    )

    return " ".join(normTweet.split())


def clean_data(df):
    '''
    Given input dataframe, creates subset of only English
    tweets, and applies text normalization according to
    normalize_tweet function above.
    '''
    print('Cleaning data & applying text normalization...\n')
    # subset of English-only tweets
    df_english = df[df['lang'] == 'en'].reset_index(drop=True)

    # text normalization
    df_english['normalized_tweet'] = df_english['full_text'].progress_apply(
        lambda tweet: normalize_tweet(tweet)
    ).str.lower()
    return df_english


def load_parquet_data(data_path, filename):
    '''
    Loads in file according to data_path and filename,
    applies necessary edits to return only English
    Tweets, and returns pandas dataframe
    '''
    # get folder name according to filename
    folder = filename.split('_')[0]

    print('Loading data from parquet file...\n')
    # load in parquet file
    df = pd.read_parquet(
        f'{data_path}/{folder}/{filename}',
    )

    # apply necessary edits
    df = clean_data(df)

    return df


def create_embedding_model(embed_model_name):
    '''
    Given string of pretrain embedding available in sentence-transformers
    library, create a SentenceTransformer object to encode embeddings with
    '''
    print('Loading SentenceTransformer...\n')
    model = SentenceTransformer(embed_model_name)
    return model


def generate_embeddings(model, tweets):
    '''
    Given a SentenceTransformer model object, and a tweets object,
    containing a pandas Series of tweets, use embedding model to
    encode tweets with model object.
    '''
    print('Generating tweet embeddings...\n')
    tweet_embeddings = model.encode(tweets, show_progress_bar=True)
    return tweet_embeddings


def generate_embedding_df(tweet_ids, tweet_embeddings):
    '''
    Given a series of tweet IDs and a list of tweet_embeddings
    (where each observation in the list is an array containing the
    embeddings), combines the two to produce a pandas DataFrame
    '''
    print('Generating pandas DataFrame with IDs and embeddings...\n')
    df = pd.DataFrame(tweet_embeddings)

    # apply more appropriate column names
    old_cols = df.columns.values
    new_cols = ['embed_' + str(x + 1) for x in old_cols]
    df.columns = new_cols

    # insert tweet IDs series as first column
    df.insert(loc=0, column='tweet_id', value=tweet_ids)

    return df


def save_embeddings(filename, embeddings_df):
    '''
    Given a filename, and two pandas dataframe, one of all the
    embeddings, and of the UMAP embeddings (reduced to 2D),
    then save the files in same location where parquet file was
    stored.
    '''
    # get folder name according to filename (i.e. the date)
    folder_date = filename.split('_')[0]
    # retrieve path to data
    data_path = path_to_data()

    # save both embedding dataframes
    embeddings_df.to_parquet(
        f'{data_path}/{folder_date}/{folder_date}_embeddings.parquet',
    )
    print('Saved embeddings to parquet file.')


def main():
    data_path = path_to_data()
    filename = str(input('What is the filename?\n'))
    EMBED_MODEL_NAME = 'distilbert-base-nli-stsb-mean-tokens'
    # load parquet file
    df = load_parquet_data(data_path, filename)
    # gather tweets
    tweets = df['normalized_tweet']
    # gather tweet IDs that are associated with first 1k tweets
    tweet_ids = df['id_str']
    # create Sentence Transformer model from distilbert
    model = create_embedding_model(EMBED_MODEL_NAME)
    # generate tweet embeddings
    tweet_embeddings = generate_embeddings(model, tweets)
    # generate df with tweet IDs and associated embedding values
    embeddings_df = generate_embedding_df(tweet_ids, tweet_embeddings)
    # save embeddings dataframe to parquet file
    save_embeddings(filename, embeddings_df)


if __name__ == '__main__':
    main()
