from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datasets import load_dataset

app = FastAPI()

dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})
intent_mapping = {intent: i for i, intent in enumerate(set(dataset["train"]["intent"]))} 
reverse_intent_mapping = {v: k for k, v in intent_mapping.items()} 

@app.get("/")
def home():
    return {"message": "Intent Detection API is running with fine-tuned model!"}

@app.post("/detect_intent/")
def detect_intent(text: str):
    model = AutoModelForSequenceClassification.from_pretrained("./trained_model", num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained("./trained_model")

    device = torch.device("cpu")
    inputs = tokenizer(text, return_tensors="pt").to(device)
    model.to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    intents = [
        {"intent": reverse_intent_mapping[i], "probability": prob.item()}
        for i, prob in enumerate(probabilities)
    ]
    intents = sorted(intents, key=lambda x: x["probability"], reverse=True)

    return {"intents": intents}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
