from sklearn.metrics import recall_score, confusion_matrix, f1_score
import pandas as pd


def get_corr_pairs(df, size):
    """

        return more correlated pairs

    """
    s = df.corr().abs().unstack().sort_values(ascending=False)
    s = s[s.values < 1]
    for i in range(size*2):
        if i % 2 == 0:
            print("{:.5f} {}".format(s[i], s.index[i]))


def get_df_uniques(df):
    """
        return a dataframe with uniques values of each column
    """
    attFeatures = []
    for col in df.columns:
        attFeatures.append(
            [col, df[col].nunique(), 
            df[col].drop_duplicates().values]
        )
    return pd.DataFrame(attFeatures, columns=['Features', 'Unique Number', 'Values'])


def show_results(test, pred):
    """
        output recall score and confusion_matrix values
    """
    recall = recall_score(test, pred, pos_label="Yes")

    print(f"recall score (Yes) : {recall}")
    cm = confusion_matrix(test, pred)
    print("True Negatives  : ", cm[0][0])
    print("False Positives : ", cm[0][1])
    print("False Negatives : ", cm[1][0])
    print("True Positives  : ", cm[1][1])
