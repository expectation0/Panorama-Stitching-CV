import cv2
import numpy as np

def donusum_matrisi_hesapla(kp1, kp2, good_matches, sol_resim, sag_resim):
    # 1. EKSİK GİDERİLDİ: Güvenlik Kilidi (En az 10 eşleşme isteyelim ki sonuç garanti olsun)
    MIN_MATCH_COUNT = 10
    
    if len(good_matches) < MIN_MATCH_COUNT:
        print(f"HATA: Yeterli eşleşme bulunamadı kanka! Gerekli: {MIN_MATCH_COUNT}, Bulunan: {len(good_matches)}")
        return None, None # Program çökmesin diye boş değer döndürüp çıkıyoruz

    # Noktaların koordinatlarını alıyoruz
    sol_noktalar = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    sag_noktalar = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC ile Homografi (Bükme Matrisi) hesaplıyoruz
    H, mask = cv2.findHomography(sol_noktalar, sag_noktalar, cv2.RANSAC, 5.0)
    
    # 2. EKSİK GİDERİLDİ: Şov Kısmı (Sadece RANSAC'ı geçenleri çizdirme)
    # mask değişkeni içinde [1, 0, 1, 1...] gibi değerler tutar. 1'ler sağlam (inlier), 0'lar çürük (outlier).
    matchesMask = mask.ravel().tolist() 

    # Sadece 1 olanları (sağlamları) yeşil renkle çizecek ayarları yapıyoruz
    draw_params = dict(matchColor=(0, 255, 0), # Sağlam ipler yeşil olsun
                       singlePointColor=None,
                       matchesMask=matchesMask, # Sadece RANSAC'ı geçenleri çiz
                       flags=2)

    # Kusursuz ipleri resim üzerine çiziyoruz
    ransac_sonrasi_resim = cv2.drawMatches(sol_resim, kp1, sag_resim, kp2, good_matches, None, **draw_params)
    
    # Yeni fotoğrafı kaydedip ekranda gösteriyoruz
    cv2.imwrite("ransac_temizlenmis_eslesmeler.jpg", ransac_sonrasi_resim)
    
    saglam_nokta_sayisi = np.sum(mask)
    print(f"RANSAC {len(good_matches)} noktadan {saglam_nokta_sayisi} tanesini kusursuz buldu. Temizlenmiş resim kaydedildi!")

    return H, mask
