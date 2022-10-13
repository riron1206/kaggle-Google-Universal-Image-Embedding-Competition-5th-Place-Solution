# kaggle Google Universal Image Embedding Competition 5th Place Solution

competition: https://www.kaggle.com/competitions/google-universal-image-embedding/

paper: https://arxiv.org/abs/

solution summary: https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/359161

## Hardware

- Ubuntu 18.04.6 LTS

- NVIDIA Teslas V100 32G / NVIDIA GeForce RTX 3090


## Software

Used the following docker image for the execution environment.
`docker pull sinpcw/pytorch:1.11.0`

- Python 3.8.12
- CUDA Version 11.3.109
- nvidia Driver Version 470.103.01

Python packages are detailed separately in requirements.txt.

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
   # (https:/www.kaggle.com/<your_kaggle_account> -> API -> Create New API Token -> kaggle.json)
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

2. Expand tfrecord and create label.csv.

   ```bash
   python tfrec2png.py
   python preprocess_glr2021_products10k.py
   
   python preprocess_GPR1200.py
   
   python preprocess_food101.py
   ```
   
## Training
1. ArcFace training

   ```bash
   python train_arcface.py
   ```

2. Classifier training

   ```bash
   python train_classifier.py
   ```

## Prediction
