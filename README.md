# cdp-paf-ml-automations
Machine learning methodology and automations for the CDP Pre-arranged Finance Report

## Setup
```
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env-example .env # Fill out .env with your HF token and OpenAI API key
```

## TF-IDF keyword exploration
```
python3 code/keyword_exploration.py
```

## Synthetic data creation (from CRS project titles and descriptions)
```
python3 code/create_synthetic_data.py
```

### Input data source
https://huggingface.co/datasets/devinitorg/cdp-paf-meta-limited

### Output data location
https://huggingface.co/datasets/devinitorg/cdp-paf-meta-limited-synthetic

## Train primary model
It is recommended you run this code in Google Colab or with a GPU.
Commented out code is left at the top for easy use with Colab.
```
python3 code/train_multiclass_weighted.py
```

### Pretrained model weights
https://huggingface.co/devinitorg/cdp-multi-classifier-weighted

## Train sub model
It is recommended you run this code in Google Colab or with a GPU.
Commented out code is left at the top for easy use with Colab.
```
python3 code/train_multiclass_sub_classes_weighted.py
```

### Pretrained model weights
https://huggingface.co/devinitorg/cdp-multi-classifier-sub-classes-weighted

## Run model inference
This relies on a CSV file of the OECD DAC CRS and outputs one file per year, e.g. crs_2022_predictions.csv.
```
python3 inference.py
```

## Run methodology automation
A python version of this script has been provided for reference, but it produces slightly different results from the original R implementation (likely due to small differences in data types, floating point errors, and regex). If you want to replicate the results from the PAF report precisely, use the R script.
This script relies on the output from inference.py.
```
Rscript code/automate.R
python3 code/automate.py
```