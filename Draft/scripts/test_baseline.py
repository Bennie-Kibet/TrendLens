from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

def test_baseline():
    model_name = "facebook/nllb-200-distilled-600M"
    print(f"Loading baseline model: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Dholuo (Luo) code in NLLB is 'luo_Latn'
    # English code is 'eng_Latn'
    translator = pipeline(
        "translation", 
        model=model, 
        tokenizer=tokenizer, 
        src_lang="eng_Latn", 
        tgt_lang="luo_Latn",
        device=-1 # Use CPU for this test
    )
    
    test_sentences = [
        "How are you today?",
        "The doctor will see you now.",
        "Take this medicine twice a day.",
        "Where does it hurt?"
    ]
    
    print("\nBaseline Translations:")
    for sentence in test_sentences:
        result = translator(sentence)
        print(f"EN: {sentence}")
        print(f"LUO: {result[0]['translation_text']}\n")

if __name__ == "__main__":
    test_baseline()
