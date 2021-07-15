"""
Jake Stephens
Class: CS 677 - Summer 2
Date: 7/13/2021
Homework #2
Description: This program makes many different calculations
to analyze the stock data for 2 tickers, SUN and SPY.
"""
import pandas as pd
import os
import math

# Ticker file location
ticker_file_sun = r"data\\SUN.csv"
ticker_file_spy = r"data\\SPY.csv"

# Function for computing True Labal
def true_label(row):
    if row["Return"] >= 0:
        return "+"
    else:
        return "-"


# ======================
# Main program execution
# ======================

# Reading csv into dataframe
df_sun = pd.read_csv(ticker_file_sun)
df_spy = pd.read_csv(ticker_file_spy)

# Grabbing SUN training subset of first 3 years
training_set_sun = df_sun.loc[df_sun["Year"] < 2019]
training_set_sun["True Label"] = training_set_sun.apply(
    lambda row: true_label(row), axis=1
)

# Grabbing SPY training subset of first 3 years
training_set_spy = df_spy.loc[df_spy["Year"] < 2018]
training_set_spy["True Label"] = training_set_spy.apply(
    lambda row: true_label(row), axis=1
)

# Function for computing the negative probability
def get_negative_probability(df, k):
    # Consecutive count
    count = 0

    # Counts of pos vs neg
    num_pos = 0
    num_neg = 0

    # Looping through each row
    for index, row in df.iterrows():
        if row["True Label"] == "-":
            if count == k:
                num_neg = num_neg + 1
                count = 0
            else:
                count = count + 1
        elif row["True Label"] == "+":
            if count == k:
                num_pos = num_pos + 1
                count = 0
            else:
                count = 0
    return num_pos, num_neg


def get_positive_probability(df, k):
    # Consecutive count
    count = 0

    # Counts of pos vs neg
    num_pos = 0
    num_neg = 0

    # Looping through each row
    for index, row in df.iterrows():
        if row["True Label"] == "+":
            if count == k:
                num_pos = num_pos + 1
                count = 0
            else:
                count = count + 1
        elif row["True Label"] == "-":
            if count == k:
                num_neg = num_neg + 1
                count = 0
            else:
                count = 0
    return num_pos, num_neg


chance_down = {
    "SUN": {
        1: {"Pos": None, "Neg": None},
        2: {"Pos": None, "Neg": None},
        3: {"Pos": None, "Neg": None},
    },
    "SPY": {
        1: {"Pos": None, "Neg": None},
        2: {"Pos": None, "Neg": None},
        3: {"Pos": None, "Neg": None},
    },
}
chance_up = {
    "SUN": {
        1: {"Pos": None, "Neg": None},
        2: {"Pos": None, "Neg": None},
        3: {"Pos": None, "Neg": None},
    },
    "SPY": {
        1: {"Pos": None, "Neg": None},
        2: {"Pos": None, "Neg": None},
        3: {"Pos": None, "Neg": None},
    },
}

# =======================================
# Populating the probability dictionaries
# =======================================

print("SUN")
print("======================")
print("Probability after seeing k consecutive 'down days'")
for k in range(1, 4):
    num_pos, num_neg = get_negative_probability(training_set_sun, k)
    print("======================")
    print("For k=" + str(k))
    print("Probability (pos:neg) = " + str(num_pos) + ":" + str(num_neg))
    chance_down["SUN"][k]["Pos"] = num_pos
    chance_down["SUN"][k]["Neg"] = num_neg

print("======================")
print("Probability after seeing k consecutive 'up days'")
for k in range(1, 4):
    num_pos, num_neg = get_positive_probability(training_set_sun, k)
    print("======================")
    print("For k=" + str(k))
    print("Probability (pos:neg) = " + str(num_pos) + ":" + str(num_neg))
    chance_up["SUN"][k]["Pos"] = num_pos
    chance_up["SUN"][k]["Neg"] = num_neg

print("SPY")
print("======================")
print("Probability after seeing k consecutive 'down days'")
for k in range(1, 4):
    num_pos, num_neg = get_negative_probability(training_set_spy, k)
    print("======================")
    print("For k=" + str(k))
    print("Probability (pos:neg) = " + str(num_pos) + ":" + str(num_neg))
    chance_down["SPY"][k]["Pos"] = num_pos
    chance_down["SPY"][k]["Neg"] = num_neg

print("======================")
print("Probability after seeing k consecutive 'up days'")
for k in range(1, 4):
    num_pos, num_neg = get_positive_probability(training_set_spy, k)
    print("======================")
    print("For k=" + str(k))
    print("Probability (pos:neg) = " + str(num_pos) + ":" + str(num_neg))
    chance_up["SPY"][k]["Pos"] = num_pos
    chance_up["SPY"][k]["Neg"] = num_neg

# =================
# Predicting Labels
# =================
print("=================================================")
print("Predicting Labels")
print("=================================================")

# Testing set of last 2 years for SUN and SPY
testing_set_sun = df_sun.loc[df_sun["Year"] > 2018]
testing_set_sun["True Label"] = testing_set_sun.apply(
    lambda row: true_label(row), axis=1
)

# Testing set of last 2 years for SUN and SPY
testing_set_spy = df_spy.loc[df_spy["Year"] > 2017]
testing_set_spy["True Label"] = testing_set_spy.apply(
    lambda row: true_label(row), axis=1
)

# Function for predicting the next label
def predict_next_label(w, df, ticker):

    predictions = []
    d = 1
    index = df.index[0]
    num_correct = 0
    num_incorrect = 0

    while index < len(df) + df.index[0]:
        # preventing key error from viewing indexs outside dataset (in beginning 3)
        if d < w:
            d = d + 1
            index = index + 1
        else:
            # generating sequence of last w labels, including current day d
            s = [None] * (w - 1)
            for i in range(w - 1):
                s[i] = df.loc[index - i]["True Label"]
            index = index + 1
            symbol = s[-1]
            num_consecutive = 1
            prediction = None
            # calculating the number of consecutive labels
            for i in range(1, (w - 1)):
                if s[-1 - i] == symbol and num_consecutive == i:
                    num_consecutive = num_consecutive + 1
            if symbol == "-":
                # compare to chance_down
                if (
                    chance_down[ticker][num_consecutive]["Pos"]
                    >= chance_down[ticker][num_consecutive]["Neg"]
                ):
                    prediction = "+"
                elif (
                    chance_down[ticker][num_consecutive]["Pos"]
                    < chance_down[ticker][num_consecutive]["Neg"]
                ):
                    prediction = "-"
            elif symbol == "+":
                # compare to chance_up
                if (
                    chance_up[ticker][num_consecutive]["Pos"]
                    >= chance_up[ticker][num_consecutive]["Neg"]
                ):
                    prediction = "+"
                elif (
                    chance_up[ticker][num_consecutive]["Pos"]
                    < chance_up[ticker][num_consecutive]["Neg"]
                ):
                    prediction = "-"
            predictions.append(prediction)
            # Preventing key error from trying to predict the next label outside the dataset
            if index + 1 < len(df.index) + df.index[0]:
                if prediction == df.loc[index + 1]["True Label"]:
                    num_correct = num_correct + 1
                else:
                    num_incorrect = num_incorrect + 1
    print("Accuracy = " + str(round((num_correct / len(df)) * 100, 2)) + "%")
    return predictions


# ===========================
# Question #2
# ===========================

# ===========================
# TESTING WITH SUN
# ===========================

print("For SUN with w=2")
sun_w_2_predictions = predict_next_label(w=2, df=testing_set_sun, ticker="SUN")
print("------------------")

print("For SUN with w=3")
sun_w_3_predictions = predict_next_label(w=3, df=testing_set_sun, ticker="SUN")
print("------------------")

print("For SUN with w=4")
sun_w_4_predictions = predict_next_label(w=4, df=testing_set_sun, ticker="SUN")
print("------------------")

# ===========================
# TESTING WITH SPY
# ===========================

print("For SPY with w=2")
spy_w_2_predictions = predict_next_label(w=2, df=testing_set_spy, ticker="SPY")
print("------------------")

print("For SPY with w=3")
spy_w_3_predictions = predict_next_label(w=3, df=testing_set_spy, ticker="SPY")
print("------------------")

print("For SPY with w=4")
spy_w_4_predictions = predict_next_label(w=4, df=testing_set_spy, ticker="SPY")
print("------------------")

# ===========================
# Question #3
# ===========================

# Method for computing the most common label for w=2,3,4
def calculate_ensemble(row):
    return row.value_counts().idxmax()

def print_ensemble_stats(ensemble_df, testing_df):

    starting_day = 3

    num_pos_correct = 0
    num_neg_correct = 0
    num_pos_incorrect = 0
    num_neg_incorrect = 0

    for index, row in ensemble_df.iterrows():
        symbol = row["ensemble"]
        if symbol == "+":
            if symbol == testing_df.iloc[starting_day + index]["True Label"]:
                num_pos_correct = num_pos_correct + 1
            else:
                num_pos_incorrect = num_pos_incorrect + 1
        else:
            if symbol == testing_df.iloc[starting_day + index]["True Label"]:
                num_neg_correct = num_neg_correct + 1
            else:
                num_neg_incorrect = num_neg_incorrect + 1
    print("Correct    (pos:neg) -- " + str(num_pos_correct) + ":" + str(num_neg_correct))
    print("Incorrect  (pos:neg) -- " + str(num_pos_incorrect) + ":" + str(num_neg_incorrect))

# Creating a new DF to hold values for ensemble learning
# I have to subsplice the lists so they will have the same lengths
ensemble_df_sun = pd.DataFrame(
    {
        "w=2": sun_w_2_predictions[2:],
        "w=3": sun_w_3_predictions[1:],
        "w=4": sun_w_4_predictions,
    }
)
ensemble_df_sun["ensemble"] = ensemble_df_sun.apply(
    lambda row: calculate_ensemble(row), axis=1
)
ensemble_df_spy = pd.DataFrame(
    {
        "w=2": spy_w_2_predictions[2:],
        "w=3": spy_w_3_predictions[1:],
        "w=4": spy_w_4_predictions,
    }
)
ensemble_df_spy["ensemble"] = ensemble_df_spy.apply(
    lambda row: calculate_ensemble(row), axis=1
)

print("== Ensemble for SUN ==")
print_ensemble_stats(ensemble_df_sun, testing_set_sun)
print_ensemble_stats(ensemble_df_spy, testing_set_spy)


# ===========================
# Question #4
# ===========================

# def calculate_stats():
#     statistics = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}









