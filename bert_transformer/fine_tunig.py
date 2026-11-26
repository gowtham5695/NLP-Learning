
import torch
import numpy as np

from datasets import Dataset, Features, Value, ClassLabel

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)


# ----------------------------------
# 1. Prepare our labelled dataset
# ----------------------------------

texts = [
    # BILLING
    "I was charged twice for my subscription.",
    "My invoice shows an incorrect amount.",
    "I need a refund for the extra payment.",
    "Why is there an additional fee on my bill?",
    "My card was charged after I cancelled.",
    "I cannot download my payment receipt.",
    "The subscription renewal amount is incorrect.",
    "My payment was deducted but the order was cancelled.",

    # DELIVERY
    "My order has not arrived yet.",
    "The package is delayed.",
    "I received the wrong delivery.",
    "When will my parcel arrive?",
    "The delivery person could not find my address.",
    "My order says delivered but I did not receive it.",
    "I want to change my delivery address.",
    "The package arrived damaged.",

    # TECHNICAL
    "The mobile app crashes when I open it.",
    "I cannot log in to my account.",
    "The website keeps showing a server error.",
    "The app freezes during checkout.",
    "I am unable to reset my password.",
    "The screen goes blank after I click submit.",
    "Notifications are not working.",
    "The application is very slow."
]

# 0 = BILLING
# 1 = DELIVERY
# 2 = TECHNICAL

labels = (
    [0] * 8 +
    [1] * 8 +
    [2] * 8
)

label_names = ["BILLING", "DELIVERY", "TECHNICAL"]

features = Features({
    "text": Value("string"),
    "labels": ClassLabel(names=label_names)
})

dataset = Dataset.from_dict(
    {
        "text": texts,
        "labels": labels
    },
    features=features
)

# Split data into training and validation sets
split = dataset.train_test_split(
    test_size=0.25,
    seed=42,
    stratify_by_column="labels"
)

train_data = split["train"]
test_data = split["test"]

print("Training samples:", len(train_data))
print("Validation samples:", len(test_data))


# ----------------------------------
# 2. Load pretrained BERT
# ----------------------------------

name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(name)

model = AutoModelForSequenceClassification.from_pretrained(
    name,
    num_labels=3,
    id2label={
        0: "BILLING",
        1: "DELIVERY",
        2: "TECHNICAL"
    },
    label2id={
        "BILLING": 0,
        "DELIVERY": 1,
        "TECHNICAL": 2
    }
)


# ----------------------------------
# 3. Tokenize the dataset
# ----------------------------------

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=64
    )


tokenized_train = train_data.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

tokenized_test = test_data.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)


# ----------------------------------
# 4. Dynamic padding
# ----------------------------------

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ----------------------------------
# 5. Evaluation metric
# ----------------------------------

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=1)

    accuracy = np.mean(predictions == labels)

    return {
        "accuracy": float(accuracy)
    }


# ----------------------------------
# 6. Training configuration
# ----------------------------------

training_args = TrainingArguments(

    output_dir="./bert_ticket_results",

    learning_rate=2e-5,

    per_device_train_batch_size=2,

    per_device_eval_batch_size=2,

    num_train_epochs=3,

    weight_decay=0.01,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="accuracy",

    save_total_limit=1,

    report_to="none"
)


# ----------------------------------
# 7. Create Trainer
# ----------------------------------

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_test,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# ----------------------------------
# 8. Fine-tune BERT
# ----------------------------------

print("Starting BERT fine-tuning...")

trainer.train()


# ----------------------------------
# 9. Evaluate the model
# ----------------------------------

results = trainer.evaluate()

print("Evaluation results:")
print(results)


# ----------------------------------
# 10. Save the fine-tuned model
# ----------------------------------

save_path = "./bert_ticket_classifier"

trainer.save_model(save_path)

tokenizer.save_pretrained(save_path)

print("Model saved successfully!")


# ----------------------------------
# 11. Test with new sentences
# ----------------------------------

from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model=save_path,
    tokenizer=save_path
)

new_sentences = [
    "I was charged extra money for my order.",
    "My parcel has not been delivered.",
    "The application crashes every time I open it."
]

predictions = classifier(new_sentences)

for sentence, prediction in zip(new_sentences, predictions):

    print("\nSentence:", sentence)
    print("Predicted category:", prediction["label"])
    print("Confidence:", prediction["score"])