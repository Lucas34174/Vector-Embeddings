import torch
from transformers import AutoTokenizer, AutoModel

MODEL = "xlm-roberta-base"
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModel.from_pretrained(MODEL).eval()

def embed_bert(textes):
    b = tok(textes, padding=True, truncation=True,
            max_length=128, return_tensors="pt")
    with torch.no_grad():
        sortie = mdl(**b).last_hidden_state      # (n, nb_tokens, dim)
    # TODO 1 : calculer le mean pooling en tenant compte de b["attention_mask"]
    # TODO 2 : normaliser chaque vecteur (norme = 1)
    return vecteurs.tolist()
