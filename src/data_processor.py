import pandas as pd
import numpy as np

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Belirtilen yoldaki CSV dosyasını yükler, sütun adlarını temizler
    ve yinelenen satırları siler.
    """
    print(f"-> Veri yükleniyor ve temizleniyor: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("HATA: Veri dosyası bulunamadı. data/ klasöründe HR_Dataset.csv dosyasının olduğundan emin olun.")
        return pd.DataFrame()

    # Sütun adlarını temizleme ve küçük harfe çevirme
    df.columns = df.columns.str.strip().str.lower()

    # Yinelenen satırları silme
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    rows_dropped = initial_rows - len(df)

    print(f"   {rows_dropped} adet mükerrer satır silindi.")

    return df