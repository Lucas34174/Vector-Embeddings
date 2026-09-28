from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

phrases = ["Le chat dort", "The cat is sleeping"]
vecs = model.encode(phrases, normalize_embeddings=True)

print(vecs[0] @ vecs[1])