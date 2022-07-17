import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("Measurement Problems/Projects/amazon_review.csv")
df.head()

# The current average rating is determined by taking the average of the ratings of the products in the current data set.
df["overall"].mean()

df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date = pd.to_datetime(df["reviewTime"].max())

df["days"] = (current_date - df["reviewTime"]).dt.days

q1 = df["days"].quantile(0.25)
q2 = df["days"].quantile(0.50)
q3 = df["days"].quantile(0.75)

 def time_based_weighted_average(dataframe, w1=29, w2=26, w3=23, w4=22):
    """
   -It is calculated in days by finding the distance of the review times from the current date.
   -The created variable "Days" is divided into quarterly values.
   -These quarterly values will be determined and the weights to be given to the time intervals will change, and a
    healthier rating process will be created by giving high weight to the current comments.

    ----------
    dataframe
    w1: If review recency is lower than or equal to the q1,w1 is used for product rating calculation.
    w2: If the review recency is greater than q1 days and equal to or less than q2 days, w2 is used for product rating calculation.
    w3: If the review recency is greater than q2 days and equal to or less than q3 days, w3 is used for product rating calculation.
    w4: If the review recency is greater than q3 days, w4 is used for product rating calculation.

    Returns
    -------

    """
    return dataframe.loc[df["days"] <= q1, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > q1) & (dataframe["days"] <= q2), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > q2) & (dataframe["days"] <= q3), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"]> q3), "overall"].mean() * w4 / 100

time_based_weighted_average(df)

#The highest weight is given for the quartile that includes recent reviews.
time_based_weighted_average(df, 29, 26, 23, 22)

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

def score_up_down_diff(helpful_yes, helpful_no): # Oran bilgisini kaçırdı.
    """
    -Reviews are voted as yes/no on whether they are helpful.
    -These votes are ranked at the top depending on which review is most helpful.
    -The score_up_down_diff function takes the difference of the up and down votes.
    -It assumes the comments with a big difference as the most helpful comments and lists them at the top for the customer to see.
    -Deficiency in here: score_up_down_diff is not satisfying, because it missed the ratio detail.

    Parameters
    ----------
    helpful_yes:int
        up count
    helpful_no: int
        down count

    Returns
    -------
    helpful_yes - helpful_no: int
    Difference between the up-down counts which can be given to the reviews.

    """
    return helpful_yes - helpful_no


def score_average_rating(helpful_yes, helpful_no): 
    """
    -It is done taking into account the ratios.
    -Deficiency in here: Frequency was not taken into account.



    Parameters
    ----------
    helpful_yes:int
        up count
    helpful_no: int
        down count

    Returns
    -------
    helpful_yes / (helpful_yes + helpful_no): float

    """
    if helpful_yes + helpful_no == 0:
        return 0
    return helpful_yes / (helpful_yes + helpful_no)

def wilson_lower_bound(helpful_yes, helpful_no, confidence=0.95):
    """
    -Calculates the Wilson Lower Bound Score.
    -The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is accepted as the WLB score.
    -The score to be calculated is used for product ranking.
    -Shows the marginal, taking into account the frequency and ratio information.

    Parameters
    ----------
    helpful_yes: int
        up count
    helpful_no: int
        down count
    confidence: float
        confidence interval for 95% confidence level

    Returns
    -------
    wilson score: float

    """
    n = helpful_yes + helpful_no
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * helpful_yes / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.head()

df.sort_values("wilson_lower_bound", ascending=False)[0:20]
