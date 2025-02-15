from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    pipeline,
    AutoTokenizer,
    AutoModelForTokenClassification,
)
from seqeval.metrics import accuracy_score
import numpy as np

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

id2label = {
    0: "0",
    1: "B-Term",
    2: "I-Term",
}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForTokenClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)


def tokenize_and_align_labels(examples):
    tokenized_inputs = []
    labels = []
    for tokens, tags in zip(examples["Tokens"], examples["Tags"]):
        bert_tokens = []
        bert_tags = []
        for i in range(len(tokens)):
            t = tokenizer.tokenize(tokens[i])
            bert_tokens += t
            bert_tags += [int(tags[i])] * len(t)
        bert_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        tokenized_inputs.append(bert_ids)
        labels.append(bert_tags)

    return {
        "input_ids": tokenized_inputs,
        "labels": labels,
    }


# Evaluation
def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [str(p) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [str(l) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = accuracy_score(true_predictions, true_labels)
    return {
        "accuracy": results,
    }


def train():
    # Download the dataset
    ds = load_dataset("thainq107/abte-restaurants")

    preprocessed_ds = ds.map(tokenize_and_align_labels, batched=True)

    # Training
    training_args = TrainingArguments(
        output_dir="./output",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=preprocessed_ds["train"],
        eval_dataset=preprocessed_ds["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()


def inference():
    token_classifier = pipeline(
        "token-classification",
        model="./output/checkpoint-1130",
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )

    text_sentences = [
        "The bread is top notch as well",
        "The service was excellent but the food was mediocre",
        "I loved the atmosphere and the decor",
        "We ordered the special, grilled branzino. It was so infused with bone that it was difficult to eat.",
    ]

    for text_sentence in text_sentences:
        results = token_classifier(text_sentence)
        print(f"Input: {text_sentence}")
        print("Extracted aspects:")
        for result in results:
            if result["entity_group"] == "Term" and result["score"] > 0.7:
                print(f"- {result['word']}: {result['entity_group']}")
        print()


if __name__ == "__main__":
    # train()
    inference()
