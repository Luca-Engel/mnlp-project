from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from huggingface_hub import login

from models.model_dpo import AutoDPOModelForCausalLM
from utils import read_json

from trl import DPOTrainer
import os


def extract_question_and_answer_pref_dataset(data):
    extracted_pref_data = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    for sample in data:
        question = sample["question_complete"]

        assert isinstance(question, str)

        for pref in sample["preference"]:
            answer_a = pref["A"]
            answer_b = pref["B"]
            overall_score = pref["overall"]

            if overall_score is None:
                continue

            a_pref = "chosen" if "A" in overall_score else "rejected"
            b_pref = "chosen" if "B" in overall_score else "rejected"

            # skip if we do not have a preference
            if a_pref == b_pref:
                continue

            extracted_pref_data["prompt"].append(question)
            extracted_pref_data[a_pref].append(answer_a)
            extracted_pref_data[b_pref].append(answer_b)

    return extracted_pref_data


hf_token = ""  # add your token here
print("logging in")
os.environ["HF_TOKEN"] = hf_token
login(hf_token)

print("Creating dataset from json file")
data = read_json("../data/M1_preference_data_15052024.json")
preference_dataset_dict = extract_question_and_answer_pref_dataset(data)

dataset_no_split = Dataset.from_dict(preference_dataset_dict)
train_test_split = dataset_no_split.train_test_split(test_size=0.2, shuffle=True, seed=0)

test_valid_split = train_test_split['test'].train_test_split(test_size=0.5, shuffle=True, seed=0)

# Combine the splits into a dictionary
dataset = {
    'train': train_test_split['train'],
    'val': test_valid_split['train'],
    'test': test_valid_split['test']
}

dataset['train'].to_json('datasets/dpo_train_dataset.jsonl', orient='records', lines=True)
dataset['val'].to_json('datasets/dpo_val_dataset.jsonl', orient='records', lines=True)
dataset['test'].to_json('datasets/dpo_test_dataset.jsonl', orient='records', lines=True)

print(dataset)

print("downloading model")
model = AutoModelForCausalLM.from_pretrained("mNLP-project/gpt2-finetuned-mcqa")

# do we not need the autoDPOModel here???
dpo_model = model  # AutoDPOModelForCausalLM(model)

print("downloading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("mNLP-project/gpt2-finetuned-mcqa")
tokenizer.pad_token = tokenizer.eos_token

# training_args = DPOConfig(
#     output_dir="./models/checkpoints",
#     beta=0.1,
# )
training_args = TrainingArguments(
    report_to="none",
    lr_scheduler_type="cosine",

    do_train=True,
    output_dir="models/checkpoints",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    learning_rate=1e-7,
    weight_decay=0.01,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    warmup_ratio=0.2,
    logging_dir="models/tenserboard",
    logging_steps=10,
    # metric_for_best_model=...,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
    seed=42,

    hub_model_id="mNLP-project/gpt2-dpo-mcqa",
)

dpo_trainer = DPOTrainer(
    model=dpo_model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    tokenizer=tokenizer,
)

tmp_trainer_for_test_set_gen = DPOTrainer(
    model=dpo_model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)
train_dataset = dpo_trainer.train_dataset
test_dataset = tmp_trainer_for_test_set_gen.eval_dataset
val_dataset = dpo_trainer.eval_dataset

train_dataset.to_json('datasets/dpo_train_dataset_from_trainer.jsonl', orient='records', lines=True)
val_dataset.to_json('datasets/dpo_val_dataset_from_trainer.jsonl', orient='records', lines=True)
test_dataset.to_json('datasets/dpo_test_dataset_from_trainer.jsonl', orient='records', lines=True)

print("pre-training evaluation validation set:")
print(dpo_trainer.evaluate())

dpo_trainer.eval_dataset = test_dataset

print("pre-training evaluation test set:")
print(dpo_trainer.evaluate())

# set back to validation set
dpo_trainer.eval_dataset = val_dataset

print("start training")
dpo_trainer.train()

print("training completed")

print("Evaluation on validation set:")
print(dpo_trainer.evaluate())

dpo_trainer.push_to_hub(
    commit_message="do test run on scitas with ref_model"
)

print("Evaluation on test set:")
dpo_trainer.eval_dataset = test_dataset

print("pre-training evaluation test set:")
print(dpo_trainer.evaluate())