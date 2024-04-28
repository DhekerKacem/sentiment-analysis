from flask import Flask, render_template, request, jsonify
from requests import HTTPError
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import os

print("init app flask")
app = Flask(__name__)
print("app flask initiated")

def load_tokenizer():
    print("Load tokenizer.")
    print(os.path.dirname(os.path.realpath(__file__)))
    tokenizer = BertTokenizer.from_pretrained('.')
    print("tokenizer loaded.")
    return tokenizer

def load_model():
    # Imaginons que c'est un modèle PyTorch
    print("Load BertModel.")
    model = BertModel.from_pretrained('.')
    print("Load BertModel.")

    print("eval model.")
    model.eval()  # Mettre le modèle en mode évaluation
    print("model evaluated.")
    return model

tokenizer = load_tokenizer()
model = load_model()

# Ajout d'une couche linéaire pour obtenir les logits
print("linear layer.")
linear_layer = nn.Linear(model.config.hidden_size, 1)
print("linear lyaer added.")


@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['tweet']
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=200,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = linear_layer(pooled_output)

    prediction = torch.sigmoid(logits).item()
    print(prediction)
    return jsonify({'sentiment': 'Positive' if prediction >= 0.75 else 'Negative'})

# define Python user-defined exceptions
class BadPredictException(Exception):
    "Raised when the prediction is not correct"
    pass

@app.route('/bad-predict', methods=['GET'])
def bad_predict():
    raise BadPredictException

@app.errorhandler(BadPredictException)
def handle_bad_predict(e):
    return 'La prédiction est mauvaise', 406

# or, without the decorator
app.register_error_handler(406, handle_bad_predict)


# Route pour la page d'accueil qui charge index.html
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run('0.0.0.0',debug=True, port=8880)  # Lancer l'application


