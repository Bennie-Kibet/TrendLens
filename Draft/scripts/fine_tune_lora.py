import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

def train():
    model_name = "facebook/nllb-200-distilled-600M"
    dataset_name = "michsethowusu/english-dholuo_sentence-pairs_mt560"
    
    # 1. Load Dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_name)
    
    # 2. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 3. Preprocessing
    def preprocess_function(examples):
        inputs = [ex for ex in examples["english"]]
        targets = [ex for ex in examples["dholuo"]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True)
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_dataset = dataset["train"].train_test_split(test_size=0.1)
    tokenized_dataset = tokenized_dataset.map(preprocess_function, batched=True)
    
    # 4. Load Model with LoRA
    print("Loading model with LoRA...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 5. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./models/nllb-dholuo-lora",
        evaluation_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True if torch.cuda.is_available() else False,
        push_to_hub=False,
    )
    
    # 6. Trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 7. Train
    print("Starting training...")
    trainer.train()
    
    # 8. Save
    model.save_pretrained("./models/nllb-dholuo-lora-final")
    print("Training complete and model saved.")

if __name__ == "__main__":
    train()
