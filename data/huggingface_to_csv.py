import numpy as np
import pandas as pd
from datasets import load_dataset

# # Clickbait-NotClickbait
# dataset = load_dataset('christinacdl/clickbait_notclickbait_dataset')
# data_train = pd.DataFrame(dataset["train"])
# data_validation = pd.DataFrame(dataset["validation"])
#
# df = pd.concat([data_train, data_validation]).reset_index(drop=True)
# df.to_csv("data/clickbait_notclickbait.csv")
#
# df = pd.read_csv("data/clickbait_notclickbait.csv", index_col=0)
# print(df)

# Ag_News
# dataset = load_dataset('ag_news')
# df = pd.DataFrame(dataset["train"])
# df = df.groupby("label").sample(n=12500, random_state=1).reset_index(drop=True)
# df.to_csv("data/ag_news.csv")
#
# df = pd.read_csv("data/ag_news.csv", index_col=0)
# print(df)

# Financial Phrasebank
# dataset = load_dataset("financial_phrasebank", "sentences_66agree")
# df = pd.DataFrame(dataset["train"])
# df = df.rename(columns={"sentence": "text"})
# df.to_csv("data/financial_phrasebank.csv")
#
# df = pd.read_csv("data/financial_phrasebank.csv", index_col=0)
# print(df)
