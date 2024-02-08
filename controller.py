from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import torch

from dotenv import load_dotenv, find_dotenv

def realizar_inferencia(prompt, tokenizer, model):
    load_dotenv(find_dotenv())
    torch.no_grad()
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, return_token_type_ids=False)
    
    # Move inputs to GPU
    # inputs = inputs.to('cuda')
    
    gc = GenerationConfig(
        # do_sample=True,
        # repetition_penalty=.9,
        # temperature=0.3,
        # top_k=10,
        # top_p=.8,
        return_full_text=False,
        min_new_tokens=5,
        # max_new_tokens=200,
        # early_stopping=True,
        # num_beams=1,
        # ngram_size_repetition_penalty=3,
        max_length=512,
        # penalty_alpha=0.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generate_ids = model.generate(
            **inputs,
            generation_config=gc,
        )

    return tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

def main():
    load_dotenv(find_dotenv())
    print("GPU Activated: ",torch.cuda.is_available())
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    # Move model to GPU
    # model.to('cuda')

    prompt = "Qual a capital do Brasil?"

    response  = realizar_inferencia(prompt, tokenizer, model)

    print(response)

if __name__ == "__main__":
    main()