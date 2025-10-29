import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import pickle

# Transformer dosyasının kaydedileceği yol
TRANSFORMER_PATH = 'models/transformer.pkl' 

def split_and_transform_data(df: pd.DataFrame):
    """
    Veri setini X (özellikler) ve y (hedef) olarak ayırır,
    özellik dönüşüm pipeline'ını (ColumnTransformer) oluşturur,
    eğitir, kaydeder ve veriyi dönüştürür.
    """
    print("\n-> Özellik mühendisliği ve dönüşümü başlatılıyor.")
    
    # 1. X ve y'yi ayırma ('left' hedef sütunu)
    # Bu aşamada X'te left sütunu hariç tüm sütunlar kalır.
    X = df.drop('left', axis=1)
    y = df['left']
    
    # 2. Transformer tanımı (Notebook'tan alınan mantık)
    # Kategorik sütunları belirleme
    cat_ordinal = ['departments', 'salary']

    # ColumnTransformer oluşturma
    column_trans = ColumnTransformer(
        # OrdinalEncoder: Kategorik verileri sıralı olarak kodlar
        [('encoder', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_ordinal)],
        # Kalan sütunlar (sayısal olanlar) için MinMaxScaler uygular
        remainder=MinMaxScaler(), 
        n_jobs=-1
    )
    
    # 3. Transformer'ı eğitme (fit) ve veriyi dönüştürme (transform)
    X_trans = column_trans.fit_transform(X)
    
    # 4. Eğitilmiş Transformer'ı kaydetme (Gelecekteki tahminler için)
    try:
        with open(TRANSFORMER_PATH, 'wb') as f:
            pickle.dump(column_trans, f)
        print(f"   Dönüşüm pipeline'ı başarıyla kaydedildi: {TRANSFORMER_PATH}")
    except Exception as e:
        print(f"HATA: Transformer kaydedilirken sorun oluştu: {e}. 'models/' klasörünü kontrol edin.")
        
    return X_trans, y