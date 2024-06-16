#https://huggingface.co/docs/transformers/en/tasks/summarization
import sys
sys.stdout = sys.__stdout__
from datasets import load_dataset

name_run = "train_5_times"
gay_epoch = 5

billsum = load_dataset("cnn_dailymail", split="ca_test")
billsum = billsum.train_test_split(test_size=0.1)
#billsum["train"][0]

from transformers import AutoTokenizer
import torch
print(torch.cuda.device_count())
if torch.cuda.device_count() > 0:
    device = 'cuda'
else:
    device = 'cpu'
print ("Running  on ",device)
checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
prefix = "summarize: "


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=3072, truncation=True)
    labels = tokenizer(text_target=examples["summary"], max_length=256, truncation=True)
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

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
training_args = Seq2SeqTrainingArguments(
    output_dir=name_run + "_output",
    eval_strategy="epoch",
    learning_rate=1e-3, #2e-5
    per_device_train_batch_size=2,
    per_device_eval_batch_size=6,
    weight_decay=0.4, #0.01
    save_total_limit=3,
    num_train_epochs=gay_epoch,
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
from transformers import TrainerCallback
class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True

trainer.add_callback(EvaluateFirstStepCallback())
trainer.train()
trainer.save_model (name_run)

#Inference
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

summarizer = pipeline("summarization", model=name_run)

def inference (text_input, device):
    text = text_input
    tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
    inputs = tokenizer(text, return_tensors="pt").input_ids.to(device)
    model = AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model").to(device)
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

text = "summarize: " + billsum["test"][0]['text']

print("summarizer (Our): \n", summarizer(text)[0]['summary_text'])
print ("_________")
print ("inference (Not our): \n",inference (text, device))
print ("_________")

text_2 = 'summarize: In economics, inflation is a general increase in the prices of goods and services in an economy. This is usually measured using the consumer price index (CPI).When the general price level rises, each unit of currency buys fewer goods and services; consequently, inflation corresponds to a reduction in the purchasing power of money.The opposite of CPI inflation is deflation, a decrease in the general price level of goods and services. The common measure of inflation is the inflation rate, the annualized percentage change in a general price index.As prices faced by households do not all increase at the same rate, the consumer price index (CPI) is often used for this purpose. Changes in inflation are widely attributed to fluctuations in real demand for goods and services (also known as demand shocks, including changes in fiscal or monetary policy), changes in available supplies such as during energy crises (also known as supply shocks), or changes in inflation expectations, which may be self-fulfilling.Moderate inflation affects economies in both positive and negative ways. The negative effects would include an increase in the opportunity cost of holding money, uncertainty over future inflation, which may discourage investment and savings, and, if inflation were rapid enough, shortages of goods as consumers begin hoarding out of concern that prices will increase in the future. Positive effects include reducing unemployment due to nominal wage rigidity,allowing the central bank greater freedom in carrying out monetary policy, encouraging loans and investment instead of money hoarding, and avoiding the inefficiencies associated with deflation. Today, most economists favour a low and steady rate of inflation.Low (as opposed to zero or negative) inflation reduces the probability of economic recessions by enabling the labor market to adjust more quickly in a downturn and reduces the risk that a liquidity trap prevents monetary policy from stabilizing the economy while avoiding the costs associated with high inflation.The task of keeping the rate of inflation low and stable is usually given to central banks that control monetary policy, normally through the setting of interest rates and by carrying out open market operations.Terminology The term originates from the Latin inflare (to blow into or inflate). Conceptually, inflation refers to the general trend of prices, not changes in any specific price. For example, if people choose to buy more cucumbers than tomatoes, cucumbers consequently become more expensive and tomatoes less expensive. These changes are not related to inflation; they reflect a shift in tastes. Inflation is related to the value of currency itself. When currency was linked with gold, if new gold deposits were found, the price of gold and the value of currency would fall, and consequently, prices of all other goods would become higher.Classical economics By the nineteenth century, economists categorised three separate factors that cause a rise or fall in the price of goods: a change in the value or production costs of the good, a change in the price of money which then was usually a fluctuation in the commodity price of the metallic content in the currency, and currency depreciation resulting from an increased supply of currency relative to the quantity of redeemable metal backing the currency. Following the proliferation of private banknote currency printed during the American Civil War, the term "inflation" started to appear as a direct reference to the currency depreciation that occurred as the quantity of redeemable banknotes outstripped the quantity of metal available for their redemption. At that time, the term inflation referred to the devaluation of the currency, and not to a rise in the price of goods.This relationship between the over-supply of banknotes and a resulting depreciation in their value was noted by earlier classical economists such as David Hume and David Ricardo, who would go on to examine and debate what effect a currency devaluation has on the price of goods.Related concepts Other economic concepts related to inflation include: deflation – a fall in the general price level;disinflation – a decrease in the rate of inflation;hyperinflation – an out-of-control inflationary spiral;stagflation – a combination of inflation, slow economic growth and high unemployment;reflation – an attempt to raise the general level of prices to counteract deflationary pressures;and asset price inflation – a general rise in the prices of financial assets without a corresponding increase in the prices of goods or services;agflation – an advanced increase in the price for food and industrial agricultural crops when compared with the general rise in prices.More specific forms of inflation refer to sectors whose prices vary semi-independently from the general trend. "House price inflation" applies to changes in the house price indexwhile "energy inflation" is dominated by the costs of oil and gas.'

print ("summarizer (Our) (text_2): \n", summarizer(text_2)[0]['summary_text'])
print ("_________")
print ("inference(Not us)(text_2): \n",inference (text_2, device))
