import pandas as pd
import numpy as np
from glob import glob
import json
import os

num_tweets = {
    "AAPL": 18365,
    "FB": 10724,
    "GOOG": 7014,
    "AMZN": 6462,
    "T": 4981,
    "D": 4411,
    "BABA": 3215,
    "MSFT": 3117,
    "PCLN": 2431,
    "BAC": 2283,
    "C": 2190,
    "JPM": 1892,
    "INTC": 1794,
    "GE": 1724,
    "DIS": 1722,
    "XOM": 1632,
    "WMT": 1552,
    "MCD": 1495,
    "PFE": 1415,
    "KO": 1388,
    "CSCO": 1358,
    "CELG": 1314,
    "V": 1186,
    "JNJ": 1173,
    "CVX": 1092,
    "CAT": 1026,
    "VZ": 1002,
    "MRK": 983,
    "WFC": 946,
    "BA": 886,
    "HD": 868,
    "AMGN": 857,
    "PG": 853,
    "ABBV": 782,
    "ORCL": 692,
    "CMCSA": 665,
    "MA": 665,
    "BP": 572,
    "UNH": 567,
    "PEP": 541,
    "MO": 499,
    "MMM": 487,
    "SLB": 459,
    "UTX": 451,
    "LMT": 410,
    "MDT": 369,
    "UPS": 331,
    "BHP": 324,
    "SO": 317,
    "NVS": 286,
    "CHTR": 284,
    "GD": 283,
    "PM": 265,
    "HON": 255,
    "EXC": 241,
    "SNY": 238,
    "CHL": 215,
    "DUK": 212,
    "NEE": 210,
    "TM": 199,
    "PCG": 198,
    "AEP": 190,
    "BUD": 188,
    "DHR": 163,
    "PPL": 162,
    "TOT": 152,
    "TSM": 118,
    "SRE": 117,
    "UN": 101,
    "UL": 92,
    "IEP": 92,
    "HSBC": 90,
    "BBL": 84,
    "REX": 81,
    "ABB": 79,
    "PTR": 56,
    "NGG": 37,
    "HRG": 35,
    "CODI": 33,
    "SNP": 31,
    "PICO": 17,
    "BRK-A": 15,
    "BSAC": 15,
    "BCH": 9,
    "SPLP": 8,
    "RDS-B": 3,
    "AGFS": 2,
}

shorted_tweets = []

for comp, tweets in num_tweets.items():
    if tweets <= 200:
        shorted_tweets.append((comp, tweets))

print("shortened")


def analyze_sentiment(text_tokens_list):
    # for each row in column, join the text tokens together into a single string
    texts = [" ".join(text_tokens) for text_tokens in text_tokens_list]
    # apply the sentiment analysis to the entire list
    results = sentiment_pipeline(texts)
    # extract labels and scores for each result
    sentiments = [(res["label"], res["score"]) for res in results]

    return sentiments


from transformers import pipeline

sentiment_pipeline = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

print("downloaded")

children = []
company_id = None
for id in range(len(shorted_tweets)):
    pid = os.fork()
    if pid == 0:
        company_id = id
        break
    children.append(pid)
    print(f"New child {pid}")

if company_id == None:
    while len(children):
        details = os.wait()
        children.pop(children.index(details[0]))
    print("\n-----\nEND PROGRAM")
else:

    tweet_df = pd.DataFrame()

    company = shorted_tweets[company_id]
    for day_f in glob(f"stock-price-predictions/tweet/{company}/*"):
        data = []
        with open(day_f, "r") as f:
            for line in f:
                data.append(json.loads(line))
        line_tweet_df = pd.DataFrame(data)
        print(line_tweet_df)
        tweet_df = pd.concat([tweet_df, line_tweet_df])

    print(tweet_df)

    sentiment_df = pd.DataFrame(
        analyze_sentiment(tweet_df["text"]), columns=["sentiment", "score"]
    )
    tweet_df = pd.concat([tweet_df, sentiment_df], axis=1)

    print(company + "-----------\n" + tweet_df)
