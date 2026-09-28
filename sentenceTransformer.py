from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

phrases = ["Le chat dort", "matory ilay saka"]
vecs = model.encode(phrases, normalize_embeddings=True)

print(vecs[0] @ vecs[1])