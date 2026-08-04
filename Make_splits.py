"""
Adds a stratified train/val/test split to dataset_index.csv.

Stratifies by (crop, disease) so every class is represented proportionally
in every split -- important here since some classes are thin (see
audit_class_balance.py). Classes with fewer than 3 images can't be split
three ways and go entirely to train, with a warning.
"""

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

VAL_FRAC = 0.15
TEST_FRAC = 0.15


def split_group(group: pd.DataFrame) -> pd.DataFrame:
    n = len(group)
    if n < 3:
        group = group.copy()
        group["split"] = "train"
        print(f"  WARNING: {group.iloc[0]['crop']} / {group.iloc[0]['disease']} "
              f"has only {n} image(s) -> all sent to train, no val/test coverage")
        return group

    train_val, test = train_test_split(group, test_size=TEST_FRAC, random_state=42)
    val_size = VAL_FRAC / (1 - TEST_FRAC)
    if len(train_val) < 2:
        train, val = train_val, train_val.iloc[0:0]
    else:
        train, val = train_test_split(train_val, test_size=val_size, random_state=42)

    train = train.copy(); train["split"] = "train"
    val = val.copy(); val["split"] = "val"
    test = test.copy(); test["split"] = "test"
    return pd.concat([train, val, test])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index_csv", nargs="?", default="dataset_index.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.index_csv)
    print(f"Splitting {len(df)} images across {df.groupby(['crop', 'disease']).ngroups} classes...")

    parts = [split_group(group) for _, group in df.groupby(["crop", "disease"])]
    df_split = pd.concat(parts, ignore_index=True)
    df_split.to_csv(args.index_csv, index=False)

    print("\nSplit sizes:")
    print(df_split["split"].value_counts().to_string())
    print("\nPer-crop split breakdown:")
    print(df_split.groupby(["crop", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()