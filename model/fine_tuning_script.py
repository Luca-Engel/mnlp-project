import torch
import numpy as np
import evaluate
import random

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_dataset, concatenate_datasets, Value, Features, Sequence, DatasetDict

# The following tutorial was used as a base for the construction of this script: 
# https://huggingface.co/docs/transformers/tasks/language_modeling

MODEL_NAME = "openai-community/gpt2"

def convert_example(example):
    question = f"Question: {example['question']}\n\nOptions:\n"
    correct_answer = example["correct_answer"]
    answer_options = [correct_answer, example["distractor1"], example["distractor2"], example["distractor3"]]
    random.shuffle(answer_options)

    for i, ans in enumerate(answer_options):
        answer_options[i] = f"{chr(65+i)}. {ans}"

    question = f"{question}" + ''.join([f'{ans}\n' for i, ans in enumerate(answer_options)]) + "\nAnswer:"

    for ans in answer_options:
        if correct_answer in ans:
            correct_answer = ans

    converted_example = {
        "title": question,
        "answers.text": correct_answer
    }
    
    return converted_example


def convert_example_old(example):
    question = example["question"]
    correct_answer = example["correct_answer"]
    answer_options = [correct_answer, example["distractor1"], example["distractor2"], example["distractor3"]]
    random.shuffle(answer_options)
    
    question = f"Which of the following is the correct answer to the question:\n{question}\n" + ''.join([f'{i+1}. {ans}\n' for i, ans in enumerate(answer_options)])

    converted_example = {
        "title": question,
        "answers.text": [correct_answer],
        "answers.score": [10],
    }
    
    return converted_example


def remove_alternative_answers(example):
    example["answers.text"] = [example["answers.text"][0]]
    example["answers.score"] = [example["answers.score"][0]]

    return example


def preprocess_function(dataset, tokenizer):
    def helper(example): # example = {'title': [all questions], 'answers.text': [all answers]}
        # example is a batch of size 1000 -> 1000 question answer pairs
        max_length = 256
        questions = example["title"] # len 1000
        answers = example["answers.text"] # len 1000

        concatenated_list = [q + " " + a for q, a in zip(questions, answers)]

        questions_tokens = tokenizer(questions, truncation=True, max_length=max_length, padding='max_length')

        with tokenizer.as_target_tokenizer():
            concatenated_tokens = tokenizer(concatenated_list, truncation=True, max_length=max_length)

        #assert False, f"conc_tokens = {concatenated_tokens.keys()}"
        questions_tokens['labels'] = concatenated_tokens['input_ids']
        return questions_tokens

    return dataset.map(helper, batched=True, remove_columns=dataset.column_names, num_proc=4)


def group_texts(examples):
    block_size = 128

    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def compute_metrics(eval_pred):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the metrics
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    
    # Extract predictions and references
    predictions, labels = eval_pred

    # Replace unknown tokens with pad tokens
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and references from token IDs to strings
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    bleu_score = bleu.compute(predictions=decoded_predictions, references=decoded_labels)
    bertscore_results = bertscore.compute(predictions=decoded_predictions, references=decoded_labels, lang='en')
    bertscore_f1 = np.mean(bertscore_results['f1'])
    bertscore_recall = np.mean(bertscore_results['recall'])
    bertscore_precision = np.mean(bertscore_results['precision'])

    # Combine metrics
    metrics = {
        'bleu': bleu_score['bleu'],
        'bertscore_precision': bertscore_precision,
        'bertscore_recall': bertscore_recall,
        'bertscore_f1': bertscore_f1,
    }

    return metrics


def compute_metrics_old(eval_pred):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the metrics
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    #comet = evaluate.load("comet")
    
    # Extract predictions and references
    predictions = eval_pred.predictions
    references = eval_pred.label_ids

    # Replace unknown tokens with pad tokens
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    references = np.where(references != -100, references, tokenizer.pad_token_id)
    
    # Decode predictions and references from token IDs to strings
    decoded_predictions = [tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True) for pred in predictions]
    decoded_references = [tokenizer.decode(ref, skip_special_tokens=True, clean_up_tokenization_spaces=True) for ref in references]
    
    # Wrap the references in a list of lists, as expected by the metrics
    decoded_references = [[ref] for ref in decoded_references]
    
    # Compute the metrics
    bleu_result = bleu.compute(predictions=decoded_predictions, references=decoded_references)
    bertscore_result = bertscore.compute(predictions=decoded_predictions, references=decoded_references, lang='en')
    #comet_result = comet.compute(predictions=decoded_predictions, references=decoded_references, model='wmt20-comet-da')
    
    # Combine the results
    metrics = {
        'bleu': bleu_result['bleu'],
        'bertscore_precision': sum(bertscore_result['precision']) / len(bertscore_result['precision']),
        'bertscore_recall': sum(bertscore_result['recall']) / len(bertscore_result['recall']),
        'bertscore_f1': sum(bertscore_result['f1']) / len(bertscore_result['f1']),
        #'comet': sum(comet_result['scores']) / len(comet_result['scores'])
    }
    
    return metrics


def preprocess_logits_for_metrics(logits, labels):
    prediction_ids = torch.argmax(logits, dim=-1)

    return prediction_ids

def upload_dataset(train, validation, test, hub_repo):
    hf_token = ""
    login(hf_token)
    dataset_dict = DatasetDict({
        "train": train,
        "validation": validation,
        "test": test
    })
    dataset_dict.push_to_hub(hub_repo)


if __name__=="__main__":
    seed = 42
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the datasets
    sciq = load_dataset("allenai/sciq")
    sciq = sciq.map(convert_example, remove_columns=['question', 'distractor1', 'distractor2', 'distractor3', 'correct_answer', 'support'])

    # Provide correct type for answers.score
    features = Features({
        "title": Value("string"),
        "answers.text": Sequence(Value('string')),
        #"answers.score": Sequence(Value("int32")),
    })
    #sciq = sciq.map(lambda example: {"answers.score": [int(value) for value in example["answers.score"]]}, features=features)

    # Shuffle and resplit eli5
    print(f"sciq len: train = {len(sciq['train'])}, val = {len(sciq['validation'])}, test = {len(sciq['test'])}")

    # Concatenate the datasets
    train_data = sciq["train"]
    val_data = sciq["validation"]
    test_data = sciq['test']
    print(f"total len after: train = {len(train_data)}, val = {len(val_data)}")

    # Upload dataset
    #upload_dataset(train_data, val_data, test_data, "mNLP-project/gpt2-finetuning")

    train_data = train_data.shuffle(seed=seed)
    val_data = val_data.shuffle(seed=seed)
    test_data = test_data.shuffle(seed=seed)

    train_len_keep = int(len(train_data) * 1)
    val_len_keep = int(len(val_data) * 1)
    train_data = train_data.select(np.arange(train_len_keep))
    val_data = val_data.select(np.arange(val_len_keep))

    # Tokenize the data
    print("Executing preprocess function...")
    train_data_tokenized = preprocess_function(train_data, tokenizer)
    val_data_tokenized = preprocess_function(val_data, tokenizer)

    train_data_tokenized = train_data_tokenized.map(lambda examples: {
        'input_ids': [ids + [tokenizer.pad_token_id] * (256 - len(ids)) for ids in examples['input_ids']],
        'attention_mask': [mask + [0] * (256 - len(mask)) for mask in examples['attention_mask']],
        'labels': [lbl + [-100] * (256 - len(lbl)) for lbl in examples['labels']]
    }, batched=True)

    val_data_tokenized = val_data_tokenized.map(lambda examples: {
        'input_ids': [ids + [tokenizer.pad_token_id] * (256 - len(ids)) for ids in examples['input_ids']],
        'attention_mask': [mask + [0] * (256 - len(mask)) for mask in examples['attention_mask']],
        'labels': [lbl + [-100] * (256 - len(lbl)) for lbl in examples['labels']]
    }, batched=True)

    # Group texts in blocks of length block_size
    print("Executing group_text function...")
    lm_train_data = train_data_tokenized.map(group_texts, batched=True, num_proc=4)
    lm_val_data = val_data_tokenized.map(group_texts, batched=True, num_proc=4)


    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Train the model
    hf_token = ""  # add your token here
    print("logging in")
    login(hf_token)
    batch_size = 8
    model_id = "mNLP-project/gpt2-finetuned-mcqa-sciq2-safety"

    training_args = TrainingArguments(
        output_dir=f"checkpoints/gpt2_finetuned_mcqa-sciq2-safety",
        do_train=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=10, 
        weight_decay=0.01,
        push_to_hub=True,
        hub_model_id=model_id,
        load_best_model_at_end=True,
        #metric_for_best_model="f1",
        seed=seed,
        report_to=[], # to avoid a weird error
        fp16=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train_data,
        eval_dataset=lm_val_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    #print(trainer.evaluate())
    trainer.train()
    print(trainer.evaluate())

    trainer.push_to_hub()
    tokenizer.push_to_hub(model_id)