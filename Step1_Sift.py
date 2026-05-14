import cv2
import os

import matcher  # onurun yazdığı matcher.py dosyasından fonksiyonları içe aktarır
import homografi
import birlestirme

# images altında üç görüntü seti: her klasörde sol + sağ kaynak fotoğraflar
GORUNTU_SETLERI = [
    ("images/Clock", "sol1.jpg", "sag1.jpg"),
    ("images/SchoolImage", "sol2.jpg", "sag2.jpg"),
    ("images/test1", "s1.jpg", "s2.jpg"),
]


def detect_features(image_path):
    # 1. Resmi Bilgisayara Oku
    # cv2.imread resmi alır. Eğer resmi bulamazsa 'None' döner.
    img = cv2.imread(image_path)

    if img is None:
        print(f"HATA: {image_path} bulunamadı! Dosya adını kontrol et.")
        return None, None, None

    # 2. Resmi Griye Çevir
    # Algoritmalar renkli resimlerle uğraşmayı sevmez, siyah-beyaz daha kolaydır.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. SIFT Dedektifini Çağır
    sift = cv2.SIFT_create()

    # 4. Anahtar Noktaları (Keypoints) ve Tanımlayıcıları (Descriptors) Bul
    # kp: Resimdeki önemli noktaların koordinatları (örn: x=150, y=200)
    # des: O noktaların kimlik kartı (2. kişinin eşleştirme yaparken kullanacağı şifreler)
    kp, des = sift.detectAndCompute(gray, None)

    # 5. Bulunan Noktaları Resmin Üzerine Çiz (Görsel Şölen)
    # DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: Noktaların büyüklüğünü ve yönünü de çizer.
    img_with_keypoints = cv2.drawKeypoints(gray, kp, img.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    print(f"{image_path} içinde {len(kp)} adet nokta bulundu!")

    return kp, des, img_with_keypoints


def _bir_klasor_isle(klasor, sol_ad, sag_ad, gorsel_goster):
    """Tek bir görüntü çifti için SIFT → eşleşme → homografi → panorama."""
    sol_resim_yolu = os.path.join(klasor, sol_ad)
    sag_resim_yolu = os.path.join(klasor, sag_ad)

    print(f"\n=== {klasor} ({sol_ad} + {sag_ad}) ===")

    kp1, des1, sol_cizimli = detect_features(sol_resim_yolu)
    kp2, des2, sag_cizimli = detect_features(sag_resim_yolu)

    if sol_cizimli is None or sag_cizimli is None:
        print("Bu set atlandı (okuma hatası).")
        return False

    cv2.imwrite(os.path.join(klasor, "sol_noktalar.jpg"), sol_cizimli)
    cv2.imwrite(os.path.join(klasor, "sag_noktalar.jpg"), sag_cizimli)

    if gorsel_goster:
        cv2.imshow("Sol Resim Noktalari", sol_cizimli)
        cv2.imshow("Sag Resim Noktalari", sag_cizimli)

    good_matches = matcher.match_features(kp1, des1, kp2, des2)

    sol_bgr_eslesme = cv2.imread(sol_resim_yolu)
    sag_bgr_eslesme = cv2.imread(sag_resim_yolu)
    img_matches = cv2.drawMatches(
        sol_bgr_eslesme,
        kp1,
        sag_bgr_eslesme,
        kp2,
        good_matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    if gorsel_goster:
        cv2.imshow("Eslesme Sonuclari", img_matches)
    cv2.imwrite(os.path.join(klasor, "eslesme_final.jpg"), img_matches)

    print("4. Aşama (Homografi ve RANSAC) hesaplanıyor...")
    H_matrisi, maske = homografi.donusum_matrisi_hesapla(
        kp1, kp2, good_matches, sol_bgr_eslesme, sag_bgr_eslesme
    )
    print("4. Aşama da tamam! Dönüşüm Matrisi (Homografi) başarıyla hesaplandı.")

    if H_matrisi is not None:
        print("5. Aşama (Birleştirme / Warping & Stitching) çalışıyor...")
        sol_bgr = cv2.imread(sol_resim_yolu)
        sag_bgr = cv2.imread(sag_resim_yolu)
        if sol_bgr is None or sag_bgr is None:
            print("HATA: Birleştirme için renkli görüntüler okunamadı.")
        else:
            panorama = birlestirme.panorama_birlestir(sol_bgr, sag_bgr, H_matrisi)
            if panorama is not None:
                cikti_yolu = os.path.join(klasor, "panorama_birlestirme.jpg")
                cv2.imwrite(cikti_yolu, panorama)
                print(f"5. Aşama tamam! Panorama kaydedildi: {cikti_yolu}")
                if gorsel_goster:
                    cv2.imshow("Panorama (5. Asama)", panorama)

    return True


# --- KODUN ÇALIŞACAĞI ANA KISIM ---
if __name__ == "__main__":
    for idx, (klasor, sol_ad, sag_ad) in enumerate(GORUNTU_SETLERI):
        # Üç seti arka arkaya işle; pencere karmaşası olmasın diye sadece son seti ekranda göster
        son_set = idx == len(GORUNTU_SETLERI) - 1
        _bir_klasor_isle(klasor, sol_ad, sag_ad, gorsel_goster=son_set)

    print("\nİşlem tamam! Üç görüntü seti işlendi. Çıkmak için herhangi bir tuşa bas.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
