import pytest
import torch
from unittest.mock import patch
from transformers import BertTokenizer, BertModel
from app.app import app

@pytest.fixture
def app_method():
    yield app

@pytest.fixture
def client(app_method):
    return app_method.test_client()

def test_home(client):
    response = client.get('/')
    assert response.status_code == 200

@patch('transformers.BertModel.from_pretrained')
@patch('transformers.BertTokenizer.from_pretrained')
@patch('torch.nn.Module.forward')  # Simuler la prédiction du modèle
def test_predict(mock_forward, mock_load_tokenizer, mock_load_model, client):
    # Setup mock
    mock_load_model.return_value = BertModel.from_pretrained('bert-base-uncased')
    mock_load_tokenizer.return_value = BertTokenizer.from_pretrained('bert-base-uncased')
    mock_forward.return_value = type('obj', (object,), {"pooler_output": torch.tensor([[0.9]])})
    
    # Payload pour la requête
    payload = {'tweet': 'so happy'}
    # Simuler une requête POST
    response = client.post('/predict', json=payload)
    
    # Vérifier les résultats
    assert response.status_code == 200
    print(response)
    data = response.get_json()
    assert data['sentiment'] == 'Positive' or 'Negative'