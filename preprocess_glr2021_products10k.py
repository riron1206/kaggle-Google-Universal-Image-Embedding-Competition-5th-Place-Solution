import os
import glob
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# =============================
# params
# =============================
DATA_NAMES = "glr2021|products10k"
N_CUT_LABEL_GLR2021 = 2000

# =============================
# DIR
# =============================
INPUT_DIR = "./data/tfrec2png"
OUTDIR = "./data"
os.makedirs(OUTDIR, exist_ok=True)

def main():
    # =============================
    # load csv
    # =============================
    df = pd.read_csv(INPUT_DIR + f"/label-noresize.csv")
    df = df[ df['path'].str.contains(DATA_NAMES) ].reset_index(drop=True)

    # =============================
    # set path
    # =============================
    paths = []
    for p in tqdm(df["path"].to_numpy()):
        paths.append( f"{INPUT_DIR}/{Path(p).parents[1].stem}/{Path(p).parents[0].stem}/{Path(p).name}")
    df["path"] = paths
    df["category"] = [Path(p).parents[1].stem for p in df["path"].to_numpy()]

    # =============================
    # cut glr2021
    # =============================
    if N_CUT_LABEL_GLR2021 > 0:
        _df = df[df["category"] != "glr2021"]
        _df_cut = df[df["category"] == "glr2021"]
        _labels = _df_cut["label"].unique()
        _labels = pd.Series(_labels).sample(n=len(_labels) - N_CUT_LABEL_GLR2021, random_state=0)
        _df_cut = _df_cut[_df_cut["label"].isin(_labels)]
        df = pd.concat([_df, _df_cut], ignore_index=True)
        df.to_csv(f"{OUTDIR}/preprocess_glr2021_products10k.csv", index=False)

if __name__ == '__main__':
    main()