"""
Challenge: AI Fashion Coordinator (Baseline For Fashion-How Challenge)

Purpose: ETRI dataset csv file train-validation split script.

Author: Daegun Kim
"""

# Import packages
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# Custom split functions
def simple_split(path:str, new_path: str):

    # Load original train set information
    df = pd.read_csv(path)
    
    # Get split
    df_tr, df_val, _, _ = train_test_split(
        df, df, test_size=0.2, random_state=42)

    # Change the column 'Split'
    df.Split.loc[df_val.index] = 'val'
    
    # Save as new csv file
    df.to_csv(new_path)


def split_each_target(path, new_path):
    # Load original train set information
    df = pd.read_csv(path)    

    # Split each target column in a stratified way
    for col in ['Daily', 'Gender', 'Embellishment']:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.2, 
            random_state=42,
        )

        for tr_idx, val_idx in sss.split(df, df[col]):
            df[col+'_Split'] = 'train'
            df[col+'_Split'].loc[val_idx] = 'val'
            break

    # Save as a new csv file
    df.to_csv(new_path_split_each_target)


# Main script
if __name__=='__main__':
    path = 'task1_data/info_etri20_emotion_train.csv'
    new_path = 'task1_data/info_etri20_emotion_tr_val_simple.csv'
    new_path_split_each_target = 'task1_data/info_etri20_emotion_tr_val_each_target.csv'

    # Simple train-validation split
    simple_split(path, new_path)

    # Stratified split for each target column
    split_each_target(path, new_path_split_each_target)

