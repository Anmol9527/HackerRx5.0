import sys
import cv2
import pytesseract
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers import pipeline
from sklearn.model_selection import train_test_split

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def extract_text_from_image(image_path):
    processed_image = preprocess_image(image_path)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed_image, config=custom_config)
    return text

def fine_tune_model(training_data, validation_data):
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
    train_encodings = tokenizer(training_data["text"].tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(validation_data["text"].tolist(), truncation=True, padding=True)
    train_labels = training_data["labels"].tolist()
    val_labels = validation_data["labels"].tolist()
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=(train_encodings, train_labels),
        eval_dataset=(val_encodings, val_labels),
    )
    trainer.train()
    return model

def extract_diagnoses(text, model=None):
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
        model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    ner_results = ner_pipeline(text)
    diagnoses = [entity['word'] for entity in ner_results if entity['entity_group'] == 'MISC']
    return diagnoses

def save_to_csv(diagnoses, output_file_path):
    df = pd.DataFrame(diagnoses, columns=["Diagnosis"])
    df.to_csv(output_file_path, index=False)

def main(input_image_path, output_file_path, fine_tune=False):
    extracted_text = extract_text_from_image(input_image_path)
    if fine_tune:
        data = pd.read_csv("labeled_data.csv")
        train_data, val_data = train_test_split(data, test_size=0.2)
        model = fine_tune_model(train_data, val_data)
    else:
        model = None
    diagnoses = extract_diagnoses(extracted_text, model)
    save_to_csv(diagnoses, output_file_path)

if __name__ == "__main__":
    input_image = sys.argv[1]
    output_file = sys.argv[2]
    fine_tune_flag = sys.argv[3] if len(sys.argv) > 3 else False
    main(input_image, output_file, fine_tune=fine_tune_flag)
