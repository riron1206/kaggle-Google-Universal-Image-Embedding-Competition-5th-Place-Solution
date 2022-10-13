import os
import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# =============================
# DIR
# =============================
INPUT_DIR = "./data/food-101/images"
OUTDIR = "./data"
os.makedirs(OUTDIR, exist_ok=True)

def main():
    df = pd.DataFrame({"path": sorted(glob.glob(f"{INPUT_DIR}/**/*.[PpJj][NnPp][GgGg]"))})
    df["label_name"] = [Path(p).parent.stem for p in df["path"].to_numpy()]
    df['label'] = LabelEncoder().fit_transform(df['label_name'].values)

    # =============================
    # limit to 30 per class
    # =============================
    _df_cat = None
    for l in sorted( df["label"].unique() ):
        _df = df[df["label"] == l]
        if len(_df) > 30:
            _df = _df.sample(n=30, random_state=0)
        if _df_cat is None:
            _df_cat = _df
        else:
            _df_cat = pd.concat([_df_cat, _df], ignore_index=True)
    df = _df_cat

    # =============================
    # create csv
    # =============================
    df['label'] = LabelEncoder().fit_transform(df['label'].values)
    df.to_csv(f"{OUTDIR}/preprocess_food101.csv", index=False)

if __name__ == '__main__':
    main()