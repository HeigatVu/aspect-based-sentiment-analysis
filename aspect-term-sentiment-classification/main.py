from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import numpy as np
import evaluate
from transformers.onnx.features import AutoModelForTokenClassification

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# Label
id2label = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}
label2id = {v: k for k, v in id2label.items()}

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased",
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)


def tokenize_and_align_labels(examples):
    sentences, sentence_tags = [], []
    labels = []
    for tokens, pols in zip(examples["Tokens"], examples["Polarities"]):
        bert_tokens = []
        bert_att = []
        pols_label = 0
        for i in range(len(tokens)):
            t = tokenizer.tokenize(tokens[i])
            bert_tokens += t
            if int(pols[i]) != -1:
                bert_att += t
                pols_label = int(pols[i])
        sentences.append(" ".join(bert_tokens))
        sentence_tags.append(" ".join(bert_att))
        labels.append(pols_label)

    tokenized_inputs = tokenizer(
        sentences, sentence_tags, padding=True, truncation=True, return_tensors="pt"
    )
    tokenized_inputs["labels"] = labels

    return tokenized_inputs


# Evaluation
def compute_metrics(pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=preprocessed_ds["train"],
        eval_dataset=preprocessed_ds["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()


def inference():
    tokenizer = AutoTokenizer.from_pretrained("./output/checkpoint-1130")

    token_classifier = pipeline(
        "token-classification",
        model="./output/checkpoint-1130",
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )

    sentiment_classifier = pipeline(
        "sentiment-analysis",
        model="./output/checkpoint-1130",
        tokenizer=tokenizer,
    )

    text_sentences = [
        "The bread is top notch as well",
        "The service was excellent but the food was mediocre",
        "I loved the atmosphere and the decor",
        "We ordered the special, grilled branzino. It was so infused with bone that it was difficult to eat.",
    ]

    for text_sentence in text_sentences:
        print(f"Input: {text_sentence}")

        # Extract aspects
        aspect_results = token_classifier(text_sentence)
        print("Extracted aspects:")
        if aspect_results:
            for result in aspect_results:
                print(f"- {result['word']} (Score: {result['score']:.4f})")

            # Combine aspects into a single string
            sentence_tags = " ".join([result["word"] for result in aspect_results])

            # Perform sentiment analysis
            sentiment_input = f"{text_sentence} [SEP] {sentence_tags}"
            sentiment_result = sentiment_classifier(sentiment_input)[0]
            print(
                f"Sentiment: {sentiment_result['label']} (Score: {sentiment_result['score']:.4f})"
            )
        else:
            print("No aspects found.")

        print()  # Add a blank line between sentences


if __name__ == "__main__":
    # train()
    inference()
