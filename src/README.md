# COVID-19 Dialogue Generation Model

## The pretrained models will be uploaded soon.

## Introduction

This is a pytorch implementation of the Chinese COVID Dialogue Generation Model.

# Environment

The code is based on python 3.7.3 and pytorch 1.4.0. The code is tested on GeForce RTX 2080Ti.

## Run

In training:

`LOAD_DIR` is the directory that you store your trained model weights

`DECODER_PATH` is the path of the trained model weights

1. First preprocess the dataset and split the data to train, validate and test sets:

   ```shell
   $ python preprocess.py
   ```

2. Train the dialogue generation model:

   ``` shell
   $ python train.py --load_dir ${LOAD_DIR}
   ```

3. Evaluation the trained dialogue generation model:

   ```shell
   $ python calculate_perplexity.py --decoder_path ${DECODER_PATH}
   ```

4. Generate responses using the trained dialogue generation model:

   ```shell
   $ python sample_generate.py --decoder_path ${DECODER_PATH}
   ```



