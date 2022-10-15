# kaggle Google Universal Image Embedding Competition 5th Place Solution

competition: https://www.kaggle.com/competitions/google-universal-image-embedding/

paper: https://arxiv.org/abs/

solution summary: https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/359161

## Hardware

- Ubuntu 18.04.6 LTS

- Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz / AMD Ryzen 9 5900X 12-Core Processor

- NVIDIA Teslas V100 32G / NVIDIA GeForce RTX 3090


## Software

Used the following docker image for the execution environment.

`docker pull sinpcw/pytorch:1.11.0`

- Python 3.8.12
- CUDA Version 11.3.109
- nvidia Driver Version 470.103.01

Python packages are detailed separately in `requirements.txt`.

## Data setup

1. Download [motono0223's google landmark recognition 2021](https://www.kaggle.com/datasets/motono0223/guie-glr2021mini-tfrecords-label-10691-17690), [motono0223's products10k](https://www.kaggle.com/datasets/motono0223/guie-products10k-tfrecords-label-1000-10690), [GPR1200](https://github.com/Visual-Computing/GPR1200), [food101](https://www.kaggle.com/datasets/srujanesanakarra/food101), [130k-images-512x512-universal-image-embeddings](https://www.kaggle.com/datasets/rhtsingh/130k-images-512x512-universal-image-embeddings) to ```./data```.

   ```bash
   apt-get update && apt-get install -y wget
   
   mkdir -p ./data
   cd ./data
   
   mkdir -p ./guie-products10k-tfrecords-label-1000-10690
   mkdir -p ./guie-glr2021mini-tfrecords-label-10691-17690
   mkdir -p ./GPR1200
   mkdir -p ./130k
   
   # Download kaggle.json from your kaggle account page and enter KAGGLE_USERNAME and KAGGLE_KEY of kaggle.json.
   # (https:/www.kaggle.com/<your_kaggle_account> -> Account -> Create New API Token -> kaggle.json)
   export KAGGLE_USERNAME=*******
   export KAGGLE_KEY=*******
   kaggle datasets download -d "motono0223/guie-products10k-tfrecords-label-1000-10690"
   kaggle datasets download -d "motono0223/guie-glr2021mini-tfrecords-label-10691-17690"
   kaggle datasets download -d "srujanesanakarra/food101"
   kaggle datasets download -d "rhtsingh/130k-images-512x512-universal-image-embeddings"
   
   wget https://visual-computing.com/files/GPR1200/GPR1200.zip
   cd ./GPR1200
   git clone https://github.com/Visual-Computing/GPR1200.git
   cd ..
   
   unzip ./guie-products10k-tfrecords-label-1000-10690.zip -d ./guie-products10k-tfrecords-label-1000-10690
   unzip ./guie-glr2021mini-tfrecords-label-10691-17690.zip -d ./guie-glr2021mini-tfrecords-label-10691-17690
   unzip ./GPR1200.zip -d ./GPR1200
   unzip ./food101.zip -d ./
   unzip ./130k-images-512x512-universal-image-embeddings.zip -d ./130k
   ```

2. Make a subset of each dataset. Also make a csv of image_path and label_id for each subset.

   - glr2021: randomly selected 5000 classes
   - products10k: all data use
   - GPR1200: delete 200 classes of iNaturalist
   - food101: randomly removed so that the maximum number of samples for each class was 30
   
   ```bash
   # convert tfrecord of glr2021 and products10k to png
   python tfrec2png.py
   # make ./data/preprocess_glr2021_products10k.csv
   python preprocess_glr2021_products10k.py
   
   # make ./data/preprocess_GPR1200.csv
   python preprocess_GPR1200.py
   
   # make ./data/preprocess_food101.csv
   python preprocess_food101.py
   ```
   
## Training
1. ArcFace model training.

   ```bash
   # make embedding and train ArcFace model
   python train_arcface.py
   ```

2. Classifier training for TTA.

   ```bash
   # make embedding of 130k dataset
   python train_classifier.py \
       emb_dir=./output/classfier_130k/embs \
       output_dir=./output/classfier_130k \
       epoch=100 \
       batch_size=512 \
       mkemb=true
   
   # train classifier
   python train_classifier.py \
       emb_dir=./output/classfier_130k/embs \
       output_dir=./output/classfier_130k \
       epoch=100 \
       batch_size=512 \
       mkemb=false
   ```

3. Make torchscript.

   ```bash
   # make ./submissions/yokoi_best_clstta.pt
   python make_torchscript_wcls.py \
       ckpt_path=./models/kqi_3090_ex072_ep650_bestlb.pth \
       ckpt_cls_path=./models/cls-last.ckpt \
       pt_path=./submissions/yokoi_best_clstta.pt
   ```

## Prediction

This competition was a code competition. Teams submitted inference notebooks which were ran on hidden test sets. We made the submission notebook on Kaggle at https://www.kaggle.com/code/hirune924/guie-submission-notebook?scriptVersionId=107598594.

Made torchscript are linked in that notebook as dataset.

