import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, StratifiedGroupKFold

def cv_split(train, seed: int, n_splits: int, cv_col: str = "label"):
    """
    StratifiedKFold
    """
    v_counts = train[cv_col].value_counts()
    v_lacks = v_counts[v_counts < n_splits].index.to_list()
    v_lacks_df = train[train[cv_col].isin(v_lacks)].reset_index(drop=True)

    train = train[~train[cv_col].isin(v_lacks)].reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, ( _, val_) in enumerate(skf.split(X=train, y=train[cv_col])):
        train.loc[val_ , "fold"] = fold
        print(f"fold{fold}:", train.loc[val_ , "fold"].shape)

    v_lacks_df["fold"] = -1
    train = pd.concat([train, v_lacks_df], ignore_index=True)

    train['fold'] = train['fold'].astype(int)
    print(train.groupby(['fold', cv_col]).size())
    return train

def cv_split_group(train, n_splits: int, cv_col: str = "label"):
    """
    GroupKFold
    """
    gkf = GroupKFold(n_splits=n_splits)
    for fold, ( _, val_) in enumerate(gkf.split(X=train, y=train[cv_col], groups=train[cv_col])):
        train.loc[val_ , "fold"] = fold
        print(f"fold{fold}:", train.loc[val_ , "fold"].shape)
    train['fold'] = train['fold'].astype(int)
    print(train.groupby(['fold', cv_col]).size())
    return train

def cv_split_stratified_group(train, seed: int, n_splits: int, cv_col: str = "category", group_col: str = "label"):
    """
    StratifiedGroupKFold
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, ( _, val_) in enumerate(sgkf.split(X=train, y=train[cv_col], groups=train[group_col])):
        train.loc[val_ , "fold"] = fold
        print(f"fold{fold}:", train.loc[val_ , "fold"].shape)
    train['fold'] = train['fold'].astype(int)
    print(train.groupby(['fold', cv_col]).size())
    print(train.groupby(['fold', group_col]).size())
    return train