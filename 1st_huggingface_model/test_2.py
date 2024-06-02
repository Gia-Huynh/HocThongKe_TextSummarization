#https://huggingface.co/docs/transformers/en/tasks/summarization
import sys
sys.stdout = sys.__stdout__
from datasets import load_dataset

billsum = load_dataset("billsum", split="ca_test")
billsum = billsum.train_test_split(test_size=0.2)
billsum["train"][0]

from transformers import AutoTokenizer
import torch
print(torch.cuda.device_count())

checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
prefix = "summarize: "


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
tokenized_billsum = billsum.map(preprocess_function, batched=True)
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

#Evaluate
import evaluate
rouge = evaluate.load("rouge")
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


#Train
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
training_args = Seq2SeqTrainingArguments(
    output_dir="my_awesome_billsum_model",
    eval_strategy="epoch",
    learning_rate=1e-4, #2e-5
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=12,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
#Inference
text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
from transformers import pipeline

summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
print(summarizer(text))

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM


def inference (text_input, model):
    text = text_input
    tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
    inputs = tokenizer(text, return_tensors="pt").input_ids
    model = AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
print (inference (text, model))

text_2 = "summarize: In economics, inflation is a general increase in the prices of goods and services in an economy. This is usually measured using the consumer price index (CPI). When the general price level rises, each unit of currency buys fewer goods and services; consequently, inflation corresponds to a reduction in the purchasing power of money. The opposite of CPI inflation is deflation, a decrease in the general price level of goods and services. The common measure of inflation is the inflation rate, the annualized percentage change in a general price index. As prices faced by households do not all increase at the same rate, the consumer price index (CPI) is often used for this purpose. Changes in inflation are widely attributed to fluctuations in real demand for goods and services (also known as demand shocks, including changes in fiscal or monetary policy), changes in available supplies such as during energy crises (also known as supply shocks), or changes in inflation expectations, which may be self-fulfilling.[10] Moderate inflation affects economies in both positive and negative ways. The negative effects would include an increase in the opportunity cost of holding money, uncertainty over future inflation, which may discourage investment and savings, and, if inflation were rapid enough, shortages of goods as consumers begin hoarding out of concern that prices will increase in the future. Positive effects include reducing unemployment due to nominal wage rigidity,[11] allowing the central bank greater freedom in carrying out monetary policy, encouraging loans and investment instead of money hoarding, and avoiding the inefficiencies associated with deflation."

print (summarizer(text_2))
print (inference (text_2, model))
