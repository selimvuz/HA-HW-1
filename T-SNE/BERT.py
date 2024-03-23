import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
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
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


# Veri setini yükle
df = pd.read_csv('../dataset/sample.csv', index_col=0)

# Talimat ve girişleri birleştir
df['soru'] = df.apply(lambda row: row['talimat'] + (' ' +
                      row['giriş'] if pd.notnull(row['giriş']) else ''), axis=1)

# Sorular ve cevaplar için vektör temsillerini al
df['soru_vektor'] = df['soru'].apply(lambda x: get_vector(x))
df['cevap_vektor'] = df['çıktı'].apply(lambda x: get_vector(x))

# Soru ve cevap vektörlerini numpy dizilerine dönüştür
soru_vektorleri = np.array(df['soru_vektor'].tolist())
cevap_vektorleri = np.array(df['cevap_vektor'].tolist())

# Vektörleri ve etiketleri birleştir
vektorler = np.vstack((soru_vektorleri, cevap_vektorleri))
etiketler = ['Soru'] * len(soru_vektorleri) + ['Cevap'] * len(cevap_vektorleri)

# Önce PCA ile boyut azaltma
pca = PCA(n_components=30)
vektorler_pca = pca.fit_transform(vektorler)

# t-SNE ile 2 boyuta indirgeme
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
vektorler_2d = tsne.fit_transform(vektorler_pca)

# Görselleştirme
plt.figure(figsize=(12, 8))
colors = {'Soru': 'blue', 'Cevap': 'orange'}
for etiket in set(etiketler):
    idx = [i for i, t in enumerate(etiketler) if t == etiket]
    plt.scatter(vektorler_2d[idx, 0], vektorler_2d[idx, 1],
                c=colors[etiket], label=etiket, alpha=0.5)
plt.legend()
plt.title('Soru/Cevap PCA -> t-SNE 2D - BERT')
plt.xlabel('')
plt.ylabel('')
plt.show()
