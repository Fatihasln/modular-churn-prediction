import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

MODEL_PATH = 'models/rf_final_model.pkl' # Kaydedilecek modelin yolu

def train_model(X_trans, y):
    """
    Veriyi eğitim ve test setine ayırır, RandomForestClassifier'ı eğitir,
    modeli kaydeder ve performans raporunu gösterir.
    """
    print("\n-> Model eğitimi başlatılıyor.")
    
    # Notebook'unuzda muhtemelen tüm veri ile eğitim yaptınız, 
    # ancak en iyi pratik için burada veriyi ayırıyoruz.
    # X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2, random_state=101, stratify=y)
    
    # Notebook'ta kullanılan model parametreleri
    model = RandomForestClassifier(n_estimators=50, max_features=4, max_depth=10, max_samples=0.8,
                                   n_jobs=-1, random_state=101, class_weight="balanced")
    
    # Modeli eğitme
    # Normalde X_train, y_train ile eğitilir. Hızlı ilerlemek için tüm veri ile eğitiyoruz:
    model.fit(X_trans, y) 
    
    # Modeli kaydetme (Transformer gibi aynı kaydetme mekanizması)
    try:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        print(f"   RandomForest modeli başarıyla kaydedildi: {MODEL_PATH}")
    except Exception as e:
        print(f"HATA: Model kaydedilirken sorun oluştu: {e}. 'models/' klasörünü kontrol edin.")
        
    # Modelin performansını raporlama (eğitim verisi üzerinde)
    y_pred = model.predict(X_trans)
    print("\n--- Model Performans Raporu (Eğitim Verisi Üzerinde) ---")
    print(classification_report(y, y_pred))
    
    return model