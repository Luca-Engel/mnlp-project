import time

import torch
import numpy as np
import random

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, \
    Trainer, pipeline, RagTokenForGeneration, RagRetriever
from datasets import load_dataset

# Use a pipeline as a high-level helper


current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sciq = load_dataset("mNLP-project/sciq_mcqa")

test_data = sciq['test']

model_names = ["openai-community/gpt2",
               "mNLP-project/gpt2-finetuned-mcqa-sciq2-safety",
               "mNLP-project/gpt2-finetuned-mcqa-sciq2-safety_rag"
               ]

rag_question_encoder = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq",
                                                             use_dummy_dataset=True).question_encoder
rag_tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True
)

MAX_NEW_TOKENS = 100


def generate_text(question, model, tokenizer):
    max_len = min(1024, len(question))

    tokens = tokenizer(question, return_tensors="pt", padding="max_length", truncation=True,
                       max_length=max_len - MAX_NEW_TOKENS)

    input_ids = tokens["input_ids"].to(current_device)
    attention_mask = tokens["attention_mask"].to(current_device)

    with torch.no_grad():
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MAX_NEW_TOKENS)

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


for model_name in model_names:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(current_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("*" * 50)
    print(f"Model: {model_name}")

    times_per_question = []
    for i in range(100):
        print(f"Question {i}")
        question = test_data[i]["question"]

        if "rag" in model_name:
            start = time.time()
            input_ids = rag_tokenizer(question, return_tensors="pt").input_ids
            question_hidden_states = rag_question_encoder(input_ids)[0]

            docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
            retrieved_docs = rag_tokenizer.batch_decode(docs_dict["context_input_ids"], skip_special_tokens=True)

            input_text = retrieved_docs[0]

            answer = generate_text(input_text, model, tokenizer)

            end = time.time()

        else:
            start = time.time()
            answer = generate_text(question, model, tokenizer)

            end = time.time()

        print(f"---------------- Answer: -------------\n{answer[0]}\n")
        print("*" * 50)

        times_per_question.append(end - start)

    with open('runtimes_rag.txt', 'a') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Average time per question: {np.mean(times_per_question)}\n")
        f.write(f"Standard deviation: {np.std(times_per_question)}\n")
        f.write(f"Max time: {np.max(times_per_question)}\n")
        f.write(f"Min time: {np.min(times_per_question)}\n")
        f.write("*" * 50)
        f.write("\n\n\n")

    print("*" * 50)
    print("\n\n\n")