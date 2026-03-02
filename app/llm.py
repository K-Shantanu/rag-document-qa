from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_answer(context, question):
    prompt = f"""
Extract the exact answer from the context.

Context:
{context}

Question: {question}

Answer with only the name. Do not add numbering. Do not add extra words.
Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    
    outputs = model.generate(
    **inputs,
    max_new_tokens=30,
    temperature=0.0,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id
)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if "." in answer:
        parts = answer.split(".")
        if len(parts) > 1:
            answer = parts[1].strip()

    answer = answer.strip()

    return answer