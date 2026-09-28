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
    mask = b["attention_mask"].unsqueeze(-1)
    sortie = (sortie * mask).sum(dim=1) / mask.sum(dim=1)
    # TODO 2 : normaliser chaque vecteur (norme = 1)
    vecteurs = torch.nn.functional.normalize(sortie, p=2, dim=1)
    return vecteurs

text_1=embed_bert(["Ny saka matory"])
text_2 = embed_bert(["Velona ilay saka"])
print(text_1[0] @ text_2[0])