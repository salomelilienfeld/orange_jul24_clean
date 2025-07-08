import pandas as pd
from IPython.display import display

df = pd.read_csv("MovieReview.csv")
display(df.head())
print(df.shape)

df = df.drop('sentiment', axis=1)

df.head()