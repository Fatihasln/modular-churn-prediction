from src.data_processor import load_and_clean_data
from src.features import split_and_transform_data
from src.model_trainer import train_model # YENİ IMPORT EKLENDİ

# Veri setinizin adını kontrol edin!
DATA_FILE_PATH = 'data/HR_Dataset.csv' 

def main():
    """
    Tüm ML iş akışını başlatan ana fonksiyon.
    """
    print("--- Müşteri Kaybı Tahmin Projesi Başlatılıyor ---")

    # Adım 1: Veriyi Yükleme ve Temizleme
    df = load_and_clean_data(DATA_FILE_PATH)

    if df.empty:
        return
    
    # Adım 2: Özellik Mühendisliği ve Dönüşümü
    X_trans, y = split_and_transform_data(df)
    
    print(f"\n--- Özellikler Dönüştürüldü ve Kaydedildi ---")
    print(f"   Dönüştürülmüş X matrisinin boyutu: {X_trans.shape}")
    print(f"   Hedef vektör y'nin boyutu: {y.shape}")
    
    # Adım 3: Model Eğitimi (YENİ ADIM)
    train_model(X_trans, y)
    
    print("\n--- İş Akışı Tamamen Tamamlandı ---")


if __name__ == "__main__":
    main()