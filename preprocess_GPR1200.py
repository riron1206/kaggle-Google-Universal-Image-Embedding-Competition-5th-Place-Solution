import os
import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# =============================
# DIR
# =============================
INPUT_DIR = "./data/GPR1200"
OUTDIR = "./data"
os.makedirs(OUTDIR, exist_ok=True)

def main():
    df = pd.DataFrame({"path": sorted(glob.glob(f"{INPUT_DIR}/images/*"))})
    df["label"] = [Path(p).stem.split("_")[0] for p in df["path"].to_numpy()]
    df["label"] = df["label"].astype(int)

    anno_df = pd.read_json(f"{INPUT_DIR}/GPR1200/GPR1200_categoryNumber_to_text.json", orient='index').reset_index()
    anno_df.columns = ["label", "name"]
    # https://github.com/Visual-Computing/GPR1200/blob/main/eval/GPR1200.py#L133
    category_list = []
    category_list += ["Landmarks"]*200
    category_list += ["iNat"]*200
    category_list += ["sketch"]*200
    category_list += ["instre"]*200
    category_list += ["sop"]*200
    category_list += ["face"]*200
    anno_df["category"] = category_list
    anno_df["label"] = anno_df["label"].astype(int)

    df = pd.merge(df, anno_df, on="label")

    # =============================
    # create csv
    # =============================
    df = df[df["category"].isin(
        [
            "Landmarks",
            "sketch",
            "instre",
            "sop",
            "face",
        ]
    )].reset_index(drop=True)
    df['label'] = LabelEncoder().fit_transform(df['label'].values)
    df.to_csv(f"{OUTDIR}/preprocess_GPR1200.csv", index=False)

if __name__ == '__main__':
    main()