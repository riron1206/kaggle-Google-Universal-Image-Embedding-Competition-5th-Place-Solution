import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.python as tfp
from tqdm import tqdm
from PIL import Image

AUTO = tf.data.experimental.AUTOTUNE

# =============================
# DIR
# =============================
INPUT_DIRS  = [
    "./data/guie-products10k-tfrecords-label-1000-10690",
    "./data/guie-glr2021mini-tfrecords-label-10691-17690",
]
OUTDIR = "./data/tfrec2png"
os.makedirs(OUTDIR, exist_ok=True)

# =============================
# func
# =============================
def deserialization_fn(serialized_example):
    parsed_example = tf.io.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
        }
    )
    image = tf.image.decode_jpeg(parsed_example['image/encoded'], channels=3)

    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(parsed_example['image/class/label'], tf.int64)
    return image, label

def rescale_image(image, label_group):
    image = tf.cast(image, tf.float32) * 255.0
    return image, label_group

def get_num_of_image(file):
    return int(file.split("/")[-1].split(".")[0].split("-")[-1])

def get_dataset(tfrecord_paths, cache=False, repeat=False, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices(tfrecord_paths)
    data_len = sum( [ get_num_of_image(file) for file in tfrecord_paths ] )
    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(deserialization_fn, num_parallel_calls=AUTO)
    dataset = dataset.map(rescale_image, num_parallel_calls = AUTO)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset

# =============================
# main
# =============================
def main():
    # =============================
    # load tfrecord
    # =============================
    train_shard_suffix = '*-train-*.tfrec'
    files = []
    for _dir in INPUT_DIRS:
        files += sorted(glob.glob(f"{_dir}/{train_shard_suffix}"))
        print(_dir, ", number of tfrecords = ", len(files))

    # =============================
    # tfrec2png
    # =============================
    count = 0
    image_path_list, label_list = [], []
    for tfrec_f in files:

        set_name = Path(tfrec_f).stem.split("-")[-4]
        f_id = Path(tfrec_f).stem

        outdir = f"{OUTDIR}/{set_name}/{f_id}"
        os.makedirs(outdir, exist_ok=True)

        ds = get_dataset([ tfrec_f ])
        for i,batch in tqdm(enumerate(ds)):

            x, y = batch[0].numpy(), batch[1].numpy()
            x, y = x[0], y[0]

            image = Image.fromarray(x.astype(np.uint8))
            image_path = f'{outdir}/{count}-{y}.png'
            image.save(image_path)

            image_path_list.append(image_path)
            label_list.append(y)
            count += 1

    df = pd.DataFrame({"path": image_path_list, "label": label_list})
    df.to_csv(f"{OUTDIR}/label-noresize.csv", index=False)

if __name__ == '__main__':
    main()