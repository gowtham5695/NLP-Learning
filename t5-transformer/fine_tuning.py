
import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_NAME = "google/flan-t5-small"

OUTPUT_DIR = "./t5_squad_finetuned"

TRAIN_SAMPLES = 10000
EVAL_SAMPLES = 1000

MAX_INPUT_LENGTH = 384
MAX_TARGET_LENGTH = 64

BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 5e-5

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# ==========================================
# 2. LOAD MODEL AND TOKENIZER
# ==========================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.to(device)

print("Model loaded successfully!")


# ==========================================
# 3. LOAD REAL DATASET
# ==========================================

print("\nDownloading SQuAD dataset...")

dataset = load_dataset("rajpurkar/squad")

train_dataset = dataset["train"].shuffle(seed=42).select(
    range(TRAIN_SAMPLES)
)

eval_dataset = dataset["validation"].shuffle(seed=42).select(
    range(EVAL_SAMPLES)
)

print("Training examples:", len(train_dataset))
print("Evaluation examples:", len(eval_dataset))


# ==========================================
# 4. PREPROCESS DATA
# ==========================================

def preprocess_function(examples):

    questions = examples["question"]
    contexts = examples["context"]
    answers = examples["answers"]

    inputs = []
    targets = []

    for question, context, answer in zip(
        questions, contexts, answers
    ):

        prompt = (
            "Answer the question using the context.\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            "Answer:"
        )

        inputs.append(prompt)

        # SQuAD contains answer text in a list.
        # Use its first available answer.
        if answer["text"]:
            targets.append(answer["text"][0])
        else:
            targets.append("")

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True
    )

    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


print("\nTokenizing training data...")

tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

print("\nTokenizing evaluation data...")

tokenized_eval = eval_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_dataset.column_names
)

print("Tokenization completed!")


# ==========================================
# 5. DATA COLLATOR
# ==========================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)


# ==========================================
# 6. TRAINING ARGUMENTS
# ==========================================

training_args = Seq2SeqTrainingArguments(

    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,

    learning_rate=LEARNING_RATE,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_steps=50,

    save_total_limit=2,

    fp16=torch.cuda.is_available(),

    report_to="none",

    predict_with_generate=True
)


# ==========================================
# 7. CREATE TRAINER
# ==========================================

trainer = Seq2SeqTrainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_eval,

    processing_class=tokenizer,

    data_collator=data_collator
)


# ==========================================
# 8. TRAIN MODEL
# ==========================================

print("\nStarting T5 fine-tuning...")

trainer.train()

print("\nTraining completed!")


# ==========================================
# 9. EVALUATE MODEL
# ==========================================

print("\nEvaluating model...")

results = trainer.evaluate()

print("Evaluation results:")
print(results)


# ==========================================
# 10. SAVE MODEL
# ==========================================

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("\nFine-tuned model saved:", OUTPUT_DIR)


# ==========================================
# 11. TEST MODEL
# ==========================================

print("\nTesting fine-tuned model...")

context = (
    "Artificial intelligence is the intelligence "
    "of machines, as opposed to the intelligence "
    "of humans and animals."
)

question = "What is artificial intelligence?"

prompt = (
    "Answer the question using the context.\n"
    f"Context: {context}\n"
    f"Question: {question}\n"
    "Answer:"
)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_INPUT_LENGTH
).to(device)

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        num_beams=4
    )

answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nQuestion:", question)
print("Generated answer:", answer)
