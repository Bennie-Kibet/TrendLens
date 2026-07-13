import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import argparse

def translate(text, model_path="facebook/nllb-200-distilled-600M"):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    translator = pipeline(
        "translation", 
        model=model, 
        tokenizer=tokenizer, 
        src_lang="eng_Latn", 
        tgt_lang="luo_Latn"
    )
    
    result = translator(text)
    return result[0]['translation_text']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate English to Dholuo")
    parser.add_argument("--text", type=str, required=True, help="Text to translate")
    parser.add_argument("--model", type=str, default="facebook/nllb-200-distilled-600M", help="Path to fine-tuned model")
    
    args = parser.parse_args()
    translation = translate(args.text, args.model)
    print(f"\nEnglish: {args.text}")
    print(f"Dholuo: {translation}")
