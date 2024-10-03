from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def top_features(vectorizer, result, n=1000):
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(result.toarray()).flatten()[::-1]

    return feature_array[tfidf_sorting][:n].tolist()

dataset = load_dataset("devinitorg/cdp-paf-meta-limited", split="train")

unrelated = dataset.filter(lambda example: 'Unrelated' in example['labels'])
cf = dataset.filter(lambda example: 'Crisis financing' in example['labels'] and not 'PAF' in example['labels'])
paf = dataset.filter(lambda example: 'PAF' in example['labels'] and not 'AA' in example['labels'])
aa = dataset.filter(lambda example: 'AA' in example['labels'])

vectorizer = TfidfVectorizer()
vectorizer.fit(dataset['text'])

unrelated_result = vectorizer.transform([" ".join(unrelated['text'])])
top_unrelated = top_features(vectorizer, unrelated_result)

cf_result = vectorizer.transform([" ".join(cf['text'])])
top_cf = top_features(vectorizer, cf_result)
top_cf = [vocab for vocab in top_cf if vocab not in top_unrelated]

paf_result = vectorizer.transform([" ".join(paf['text'])])
top_paf = top_features(vectorizer, paf_result)
top_paf = [vocab for vocab in top_paf if vocab not in top_unrelated]
top_paf = [vocab for vocab in top_paf if vocab not in top_cf]


aa_result = vectorizer.transform([" ".join(aa['text'])])
top_aa = top_features(vectorizer, aa_result)
top_aa = [vocab for vocab in top_aa if vocab not in top_unrelated]
top_aa = [vocab for vocab in top_aa if vocab not in top_cf]
top_aa = [vocab for vocab in top_aa if vocab not in top_paf]
