from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
import math


global TOKENIZER
global DEVICE
global MODEL
TOKENIZER = AutoTokenizer.from_pretrained('devinitorg/ODABert', model_max_length=512)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL = AutoModelForSequenceClassification.from_pretrained("devinitorg/cdp-multi-classifier-weighted")
MODEL = MODEL.to(DEVICE)

SUB_MODEL = AutoModelForSequenceClassification.from_pretrained("devinitorg/cdp-multi-classifier-sub-classes-weighted")
SUB_MODEL = SUB_MODEL.to(DEVICE)

YEAR = 2022


def sigmoid(x):
   return 1/(1 + np.exp(-x))


def chunk_by_tokens(input_text, model_max_size=512):
    chunks = list()
    tokens = TOKENIZER.encode(input_text)
    token_length = len(tokens)
    if token_length <= model_max_size:
        return [input_text]
    desired_number_of_chunks = math.ceil(token_length / model_max_size)
    calculated_chunk_size = math.ceil(token_length / desired_number_of_chunks)
    for i in range(0, token_length, calculated_chunk_size):
        chunks.append(TOKENIZER.decode(tokens[i:i + calculated_chunk_size]))
    return chunks


def inference(model, inputs):
    predictions = model(**inputs)

    logits = predictions.logits.cpu().detach().numpy()[0]
    predicted_confidences = sigmoid(logits)
    predicted_classes = (predicted_confidences > 0.5)

    return predicted_classes, predicted_confidences

def map_columns(example):
    textual_data_list = [
        example['project_title'],
        example['short_description'],
        example['long_description']
    ]
    textual_data_list = [str(textual_data) for textual_data in textual_data_list if textual_data is not None]
    text = " ".join(textual_data_list)

    predictions = {
        'Crisis finance': [False, 0],
        'PAF': [False, 0],
        'AA': [False, 0],
        'Direct': [False, 0],
        'Indirect': [False, 0],
        'Part': [False, 0],
    }

    text_chunks = chunk_by_tokens(text)
    for text_chunk in text_chunks:
        inputs = TOKENIZER(text_chunk, return_tensors="pt", truncation=True).to(DEVICE)
        model_pred, model_conf = inference(MODEL, inputs)
        predictions['Crisis finance'][0] = predictions['Crisis finance'][0] or model_pred[0]
        predictions['Crisis finance'][1] = max(predictions['Crisis finance'][1], model_conf[0])
        predictions['PAF'][0] = predictions['PAF'][0] or model_pred[1]
        predictions['PAF'][1] = max(predictions['PAF'][1], model_conf[1])
        predictions['AA'][0] = predictions['AA'][0] or model_pred[2]
        predictions['AA'][1] = max(predictions['AA'][1], model_conf[2])

        # Apply sub-model
        sub_model_pred, sub_model_conf = inference(SUB_MODEL, inputs)
        predictions['Direct'][0] = predictions['Direct'][0] or sub_model_pred[0]
        predictions['Direct'][1] = max(predictions['Direct'][1], sub_model_conf[0])
        predictions['Indirect'][0] = predictions['Indirect'][0] or sub_model_pred[1]
        predictions['Indirect'][1] = max(predictions['Indirect'][1], sub_model_conf[1])
        predictions['Part'][0] = predictions['Part'][0] or sub_model_pred[2]
        predictions['Part'][1] = max(predictions['Part'][1], sub_model_conf[2])

    example['Crisis finance predicted'] = predictions['Crisis finance'][0]
    example['Crisis finance confidence'] = predictions['Crisis finance'][1]
    example['PAF predicted'] = predictions['PAF'][0]
    example['PAF confidence'] = predictions['PAF'][1]
    example['AA predicted'] = predictions['AA'][0]
    example['AA confidence'] = predictions['AA'][1]
    example['Direct predicted'] = predictions['Direct'][0]
    example['Direct confidence'] = predictions['Direct'][1]
    example['Indirect predicted'] = predictions['Indirect'][0]
    example['Indirect confidence'] = predictions['Indirect'][1]
    example['Part predicted'] = predictions['Part'][0]
    example['Part confidence'] = predictions['Part'][1]
    return example

def main():
    text_cols = ['project_title', 'short_description', 'long_description']
    dataset = pd.read_csv("crs_{}.csv".format(YEAR))
    dataset_text = dataset[text_cols]
    dataset_text = Dataset.from_pandas(dataset_text)
    dataset_text = dataset_text.map(map_columns, remove_columns=text_cols)
    dataset_text = pd.DataFrame(dataset_text)
    dataset = pd.concat([dataset.reset_index(drop=True), dataset_text.reset_index(drop=True)], axis=1)
    dataset.to_csv('crs_{}_predictions.csv'.format(YEAR), index=False)


if __name__ == '__main__':
    main()

