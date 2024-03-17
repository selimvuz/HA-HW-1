import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

N = 1

# Hugging Face modelini yükle
print("Model yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/turkish-gpt2-medium")
model = AutoModel.from_pretrained("ytu-ce-cosmos/turkish-gpt2-medium")

# CUDA kullanılabilirse modeli GPU'ya taşı
if torch.cuda.is_available():
    model = model.cuda()
    print("Model GPU'ya taşındı.")

# Tokenizer için padding token ayarla
tokenizer.pad_token = tokenizer.eos_token

def get_vector(text):
    global N
    # Metni modele göre tokenize et ve vektör temsilini al
    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True, max_length=512)
    # CUDA kullanılabilirse, tensor'ları GPU'ya taşı
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"{N}/103.126 metin vektöre dönüştürüldü.")
    N += 1
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Veri setini yükle
print("Veri kümesi yükleniyor...")
df = pd.read_csv('../dataset/instructions.csv', index_col=0)

print("Talimat ve girişler birleştiriliyor...")
# Talimat ve girişleri birleştir
df['soru'] = df.apply(lambda row: row['talimat'] + (' ' +
                      row['giriş'] if pd.notnull(row['giriş']) else ''), axis=1)

# Sorular ve cevaplar için vektör temsillerini al
print("Vektör temsilleri alınıyor...")
df['soru_vektor'] = df['soru'].apply(lambda x: get_vector(x))
df['cevap_vektor'] = df['çıktı'].apply(lambda x: get_vector(x))

# Rastgele 1000 soru seç
print("Rastgele 1000 cevap seçiliyor...")
sample_answers = df.sample(n=1000, random_state=42)

# Cevaplara göre değerlendirme yap
top1_success = 0
top5_success = 0

N = 1  # İşlem sırasını takip etmek için
for _, row in sample_answers.iterrows():
    print(f"{N}/1000 cevap için en yakın sorular hesaplanıyor...")
    N += 1
    answer_vector = row['cevap_vektor']
    similarities = df.apply(lambda x: cosine_similarity(
        answer_vector, x['soru_vektor'])[0][0], axis=1)
    sorted_similarities = similarities.sort_values(ascending=False)

    # Top1 ve Top5 başarısını ölç
    if row.name == sorted_similarities.index[0]:
        top1_success += 1
    if row.name in sorted_similarities.index[:5]:
        top5_success += 1

print(f"Top1 Başarısı: {top1_success / len(df)}")
print(f"Top5 Başarısı: {top5_success / len(df)}")
