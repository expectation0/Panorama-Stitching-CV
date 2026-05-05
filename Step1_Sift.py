import cv2
import os


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


# --- KODUN ÇALIŞACAĞI ANA KISIM ---
if __name__ == "__main__":
    # Resimlerin yolları (images klasörü içindeki resimler)
    sol_resim_yolu = "images/Clock/sol1.jpg"
    sag_resim_yolu = "images/Clock/sag1.jpg"

    # Sol resim için fonksiyonu çalıştır
    kp1, des1, sol_cizimli = detect_features(sol_resim_yolu)

    # Sağ resim için fonksiyonu çalıştır
    kp2, des2, sag_cizimli = detect_features(sag_resim_yolu)

    # Eğer resimler başarıyla işlendiyse, sonuçları kaydet ve ekranda göster
    if sol_cizimli is not None and sag_cizimli is not None:
        # Sonuçları klasöre yeni resim olarak kaydet (GitHub'daki arkadaşlarının görmesi için)
        cv2.imwrite("images/Clock/sol_noktalar.jpg", sol_cizimli)
        cv2.imwrite("images/Clock/sag_noktalar.jpg", sag_cizimli)

        # Ekranda göster
        cv2.imshow("Sol Resim Noktalari", sol_cizimli)
        cv2.imshow("Sag Resim Noktalari", sag_cizimli)

        print("İşlem tamam! Sonuçlar ekranda. Çıkmak için herhangi bir tuşa bas.")

        # Klavyeden bir tuşa basılana kadar resimleri ekranda tutar, sonra pencereleri kapatır
        cv2.waitKey(0)
        cv2.destroyAllWindows()