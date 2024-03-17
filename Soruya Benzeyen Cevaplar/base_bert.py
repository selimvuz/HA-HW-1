import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# Hugging Face BERT modelini yükle
tokenizer = AutoTokenizer.from_pretrained(
    "ytu-ce-cosmos/turkish-base-bert-uncased")
model = AutoModel.from_pretrained("ytu-ce-cosmos/turkish-base-bert-uncased")

# CUDA kullanılabilirse modeli GPU'ya taşı
if torch.cuda.is_available():
    model = model.cuda()

def get_vector(text):
    # Metni modele göre tokenize et ve vektör temsilini al
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True, max_length=512)
    # CUDA kullanılabilirse, tensor'ları GPU'ya taşı
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Sonuçları CPU'ya geri taşı ve numpy array'ine çevir
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Veri setini yükle
df = pd.read_csv('../dataset/instructions.csv', index_col=0)

# Talimat ve girişleri birleştir
df['soru'] = df.apply(lambda row: row['talimat'] + (' ' +
                      row['giriş'] if pd.notnull(row['giriş']) else ''), axis=1)

# Sorular ve cevaplar için vektör temsillerini al
df['soru_vektor'] = df['soru'].apply(lambda x: get_vector(x))
df['cevap_vektor'] = df['çıktı'].apply(lambda x: get_vector(x))

# Rastgele 1000 soru seç
sample_questions = df.sample(n=1000, random_state=42)

top1_success = 0
top5_success = 0

for _, row in sample_questions.iterrows():
    question_vector = row['soru_vektor']
    similarities = df.apply(lambda x: cosine_similarity(
        question_vector.reshape(1, -1), x['cevap_vektor'].reshape(1, -1))[0][0], axis=1)
    sorted_similarities = similarities.sort_values(ascending=False)

    # Top1 ve Top5 başarıyı kontrol et
    if row.name == sorted_similarities.index[0]:
        top1_success += 1
    if row.name in sorted_similarities.index[:5]:
        top5_success += 1

print(f"Top1 Başarısı: {top1_success / 1000}")
print(f"Top5 Başarısı: {top5_success / 1000}")
