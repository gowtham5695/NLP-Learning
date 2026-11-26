
# ============================================================
# ROBERTA FINE-TUNING FOR IMDB SENTIMENT CLASSIFICATION
# WITH AUTOMATIC CHECKPOINT SAVING AND RESUMING
# ============================================================

# 1. IMPORT LIBRARIES

import os
import torch
import numpy as np

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

from transformers.trainer_utils import get_last_checkpoint

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

MODEL_NAME = "FacebookAI/roberta-base"

SAVE_PATH = "./my_finetuned_roberta"

OUTPUT_DIR = "./roberta_imdb_results"

MAX_LENGTH = 256

BATCH_SIZE = 4

EPOCHS = 2

SAVE_STEPS = 500

print("Using device:", "CUDA" if torch.cuda.is_available() else "CPU")

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. CHECK FOR EXISTING CHECKPOINT
# ============================================================

print("\nChecking for existing checkpoints...")

last_checkpoint = None

if os.path.isdir(OUTPUT_DIR):

    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

if last_checkpoint is not None:

    print("Checkpoint found!")
    print("Resuming from:", last_checkpoint)

else:

    print("No checkpoint found.")
    print("Starting fresh training.")


# ============================================================
# 4. LOAD PRETRAINED ROBERTA TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ============================================================
# 5. LOAD ROBERTA WITH CLASSIFICATION HEAD
# ============================================================

print("\nLoading pretrained RoBERTa model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "NEGATIVE",
        1: "POSITIVE"
    },
    label2id={
        "NEGATIVE": 0,
        "POSITIVE": 1
    }
)

print(model.config)


# ============================================================
# 6. LOAD IMDB DATASET
# ============================================================

print("\nLoading IMDb dataset...")

dataset = load_dataset("stanfordnlp/imdb")

print(dataset)

print("\nSample review:")
print(dataset["train"][0])


# ============================================================
# 7. TOKENIZATION
# ============================================================

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )


print("\nTokenizing dataset...")

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)


# ============================================================
# 8. PREPARE TRAINING AND TESTING DATA
# ============================================================

train_dataset = tokenized_dataset["train"]

test_dataset = tokenized_dataset["test"]

print("\nTraining samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))


# ============================================================
# 9. DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# 10. EVALUATION METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# 11. TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    # Evaluate after each epoch
    eval_strategy="epoch",

    # Save checkpoints every 500 training steps
    save_strategy="steps",

    save_steps=SAVE_STEPS,

    # Keep only the latest 2 checkpoints
    save_total_limit=2,

    learning_rate=2e-5,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,

    weight_decay=0.01,

    logging_steps=100,

    # GTX 1650: keep FP16 disabled
    fp16=False,

    report_to="none",

    # Evaluation and checkpoint saving use different schedules,
    # so best-model loading is disabled here.
    load_best_model_at_end=False,

    # Save optimizer and scheduler states for resuming
    save_only_model=False
)


# ============================================================
# 12. CREATE TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# ============================================================
# 13. TRAIN OR RESUME ROBERTA
# ============================================================

print("\nStarting RoBERTa fine-tuning...")

if last_checkpoint is not None:

    print("\nResuming previous training checkpoint...")

    trainer.train(
        resume_from_checkpoint=last_checkpoint
    )

else:

    print("\nStarting a new training session...")

    trainer.train()


# ============================================================
# 14. EVALUATE MODEL
# ============================================================

print("\nEvaluating model...")

results = trainer.evaluate()

print("\nEvaluation Results:")

for key, value in results.items():

    print(key, ":", value)


# ============================================================
# 15. PREDICTION FUNCTION
# ============================================================

def predict_sentiment(review):

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    device = next(trainer.model.parameters()).device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    trainer.model.eval()

    with torch.no_grad():

        outputs = trainer.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

        prediction = torch.argmax(
            probabilities,
            dim=-1
        ).item()

    label = trainer.model.config.id2label[prediction]

    confidence = probabilities[0][prediction].item()

    return label, confidence


# ============================================================
# 16. TEST WITH NEW REVIEWS
# ============================================================

reviews = [

    "This movie was absolutely fantastic. I loved it!",

    "The movie was boring and a complete waste of time.",

    "The acting was brilliant and the story was wonderful."

]

print("\nTesting predictions...")

for review in reviews:

    label, confidence = predict_sentiment(review)

    print("\nReview:", review)

    print("Predicted Sentiment:", label)

    print("Confidence:", round(confidence * 100, 2), "%")


# ============================================================
# 17. SAVE FINE-TUNED MODEL
# ============================================================

print("\nSaving fine-tuned model...")

trainer.save_model(SAVE_PATH)

tokenizer.save_pretrained(SAVE_PATH)

print("Model saved successfully!")


# ============================================================
# 18. RELOAD SAVED MODEL
# ============================================================

print("\nReloading fine-tuned model...")

loaded_tokenizer = AutoTokenizer.from_pretrained(
    SAVE_PATH
)

loaded_model = AutoModelForSequenceClassification.from_pretrained(
    SAVE_PATH
)

loaded_model.eval()

print("Fine-tuned RoBERTa loaded successfully!")


# ============================================================
# 19. FINAL VERIFICATION
# ============================================================

print("\nModel configuration:")

print("Architecture:", loaded_model.config.architectures)

print("Number of labels:", loaded_model.config.num_labels)

print("Labels:", loaded_model.config.id2label)

print("\nROBERTA FINE-TUNING COMPLETED SUCCESSFULLY!")
