import torch
import numpy as np
import evaluate
import random

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, \
    Trainer
from datasets import load_dataset, concatenate_datasets, Value, Features, Sequence, DatasetDict


def convert_example(example):
    question = f"Question: {example['question']}\n\nOptions:\n"
    correct_answer = example["correct_answer"]
    answer_options = [correct_answer, example["distractor1"], example["distractor2"], example["distractor3"]]
    random.shuffle(answer_options)

    scores = [10 if ans == correct_answer else 1 for ans in answer_options]
    scores = [int(score) for score in scores]

    for i, ans in enumerate(answer_options):
        answer_options[i] = f"{chr(65 + i)}. {ans}"

    question = f"{question}" + ''.join([f'{ans}\n' for i, ans in enumerate(answer_options)]) + "\nAnswer:"

    max_index = scores.index(max(scores))
    as_letter = chr(65 + max_index)

    converted_example = {
        "title": question,
        "answers.text": answer_options,
        "answers.score": scores,
        "question": question,
        "answer": as_letter,
    }

    return converted_example


sciq = load_dataset("allenai/sciq")
sciq = sciq.map(convert_example,
                remove_columns=['question', 'distractor1', 'distractor2', 'distractor3', 'correct_answer', 'support'])

# Provide correct type for answers.score
features = Features({
    "title": Value("string"),
    "answers.text": Sequence(Value('string')),
    "answers.score": Sequence(Value("int32")),
    "question": Value("string"),
    "answer": Value("string"),
})
sciq = sciq.map(lambda example: {"answers.score": [int(value) for value in example["answers.score"]]},
                features=features)

print(f"sciq len: train = {len(sciq['train'])}, val = {len(sciq['validation'])}, test = {len(sciq['test'])}")

# Concatenate the datasets
train_data = sciq["train"]
val_data = sciq["validation"]
test_data = sciq['test']

dataset_dict = DatasetDict({
    "train": train_data,
    "validation": val_data,
    "test": test_data
})

hf_token = ""  # add your token here
print("logging in")
login(hf_token)

dataset_dict.push_to_hub("mNLP-project/sciq_mcqa")
