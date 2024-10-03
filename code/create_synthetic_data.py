import os
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
import click
from datasets import load_dataset, concatenate_datasets, Dataset
from tqdm import tqdm
from huggingface_hub import login


load_dotenv()
client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)
HF_TOKEN = os.getenv('HF_TOKEN')
login(token=HF_TOKEN)

MODEL = "gpt-3.5-turbo-0125"
MULTIPLIER = 10


def warn_user_about_tokens(tokenizer, text):
    token_cost = 0.5
    cost_per = 1000000
    token_count = len(tokenizer.encode(text))
    return click.confirm(
        "This will use at least {} tokens and cost at least ${} to run. Do you want to continue?".format(
        token_count, round((token_count / cost_per) * token_cost, 4)
    )
    , default=False)

if __name__ == '__main__':

    dataset = load_dataset("devinitorg/cdp-paf-meta-limited", split="train")

    # Remove unrelated and keep only positive samples
    dataset = dataset.filter(lambda example: example["labels"] not in ["Unrelated", "Crisis financing"])

    def relabel_meta_to_multi(example):
        all_labels = example['labels'].split(',')
        if len(all_labels) > 1 and 'Crisis financing' in all_labels:
            all_labels.remove('Crisis financing')
        example['labels'] = ','.join(all_labels)
        return example

    dataset = dataset.map(relabel_meta_to_multi, num_proc=8)

    dataset = dataset.add_column("class_labels", dataset['labels'])

    dataset = dataset.class_encode_column('class_labels').train_test_split(
        test_size=0.7,
        stratify_by_column="class_labels",
        shuffle=True,
        seed=42
    )

    dataset = dataset.remove_columns(["class_labels"])
    dataset_train = dataset['train']

    # format (Symantic description of categories, extra instructions)
    symantic_label_mapping = {
        'PAF,Direct': ('Direct Pre-Arranged Financing for Crises', 'without using those words in the record'),
        'PAF,Indirect': ('Indirect Pre-Arranged Financing for Crises','without using those words in the record'),
        'PAF,Part': ('Partial Pre-Arranged Financing for Crises','without using those words in the record'),
        'PAF,Direct,Indirect': ('Both Direct and Indirect Pre-Arranged Financing for Crises','without using those words in the record'),
        'PAF,Direct,AA': ('Direct Pre-Arranged Financing for Crises and Direct Anticipatory Action for Humanitarian Crises','without using those words in the record'),
        'PAF,Indirect,AA': ('Indirect Pre-Arranged Financing for Crises and Indirect Anticipatory Action for Humanitarian Crises','without using those words in the record'),
        'PAF,Direct,Contingent financing': ('Direct Pre-Arranged Financing for Crises and Contingent Financing','without using "pre-arranged" in the record. You may use "contingent" or "contingent finance" in the record'),
        'PAF,Direct,Indirect,AA': ('Both Direct and Indirect Pre-Arranged Financing for Crises and Direct and Indirect Anticipatory Action for Humanitarian Crises','without using those words in the record'),
        'PAF,Direct,WB CAT DDO,Contingent financing': ('Direct Pre-Arranged Financing for Crises, World Bank Catastrophe Deferred Drawdown Option, and Contingent Financing','without using "pre-arranged" in the record. You may use "contingent", "contingent finance", "CAT DDO", or "Catastrophe Deferred Drawdown Option" in the record. You must use "International Bank for Reconstruction and Development" or "International Development Association" in the record.'),
        'PAF,Part,AA': ('Partial Pre-Arranged Financing for Crises and Partial Anticipatory Action for Humanitarian Crises','without using those words in the record'),
        'PAF,Indirect,Contingent financing': ('Indirect Pre-Arranged Financing for Crises and Contingent Financing','without using "pre-arranged" in the record. You may use "contingent" o "contingent finance" in the record'),
    }
    system_prompt_format = "Below is a record from a database of development and humanitarian assistance. I need your help to create synthetic data to train a classifier network. Could you please write {} synthetic records based on the example, separated by new lines, that mirrors it in length, content, vocabulary, language, and theme? The synthetic record should reflect the themes we are trying to classify, which are '{}' {}. Please only write the synthetic data and no additional text."

    def apply_system_prompts(example):
        categories, extra_instructions = symantic_label_mapping[example["labels"]]
        example["system_prompt"] = system_prompt_format.format(MULTIPLIER, categories, extra_instructions)
        return example
    
    dataset_train = dataset_train.map(apply_system_prompts, num_proc=8)
    all_prompts = " ".join(dataset_train["system_prompt"])
    dataset_texts = " ".join(dataset_train["text"] * (MULTIPLIER + 1))
    all_text = all_prompts + dataset_texts
    tokenizer = tiktoken.encoding_for_model(MODEL)

    if warn_user_about_tokens(tokenizer, text=all_text) == True:
        synthetic_labels = list()
        synthetic_texts = list()
        for i, user_prompt in tqdm(enumerate(dataset_train["text"]), total=dataset_train.num_rows):
            system_prompt = dataset_train["system_prompt"][i]
            label = dataset_train["labels"][i]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages
                )
                for synthetic_text in response.choices[0].message.content.split("\n"):
                    if synthetic_text != '':
                        synthetic_texts.append(synthetic_text)
                        synthetic_labels.append(label)
            except:
                print("Error fetching result {} from OpenAI.".format(i))

        synthetic_dataset = Dataset.from_dict({
            'text': synthetic_texts,
            'labels': synthetic_labels
        })
        dataset['train'] = concatenate_datasets([dataset['train'], synthetic_dataset])
        dataset.push_to_hub("devinitorg/cdp-paf-meta-limited-synthetic")
