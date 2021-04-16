from sklearn.metrics import recall_score, confusion_matrix, f1_score
import pandas as pd

def get_df_uniques(df):
    attFeatures = []
    for col in df.columns:
        attFeatures.append(
            [col, df[col].nunique(), 
            df[col].drop_duplicates().values]
        )
    return pd.DataFrame(attFeatures, columns=['Features', 'Unique Number', 'Values'])


def show_results(test, pred):
    recall = recall_score(test, pred, pos_label="Yes")

    print(f"recall score (Yes) : {recall}")
    cm = confusion_matrix(test, pred)
    print("True Negatives  : ", cm[0][0])
    print("False Positives : ", cm[0][1])
    print("False Negatives : ", cm[1][0])
    print("True Positives  : ", cm[1][1])

