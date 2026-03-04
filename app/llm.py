from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_answer(context, question):
    prompt = f"""
    You are answering questions about a document.

    Use only the context below to answer the question.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    
    outputs = model.generate(
    **inputs,
    max_new_tokens=50,
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