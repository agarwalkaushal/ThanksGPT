from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = FastAPI()

intent_mapping = {'non_airtel': 0, 'safe_custody': 1, 'non_safe_custody': 2}
reverse_intent_mapping = {0: 'non_airtel', 1: 'safe_custody', 2: 'non_safe_custody'}

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
    print(probabilities)
    intents = [
        {"intent": reverse_intent_mapping[i], "probability": prob.item()}
        for i, prob in enumerate(probabilities)
    ]
    intents = sorted(intents, key=lambda x: x["probability"], reverse=True)

    return {"intents": intents}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
