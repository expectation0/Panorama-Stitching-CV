import cv2
import numpy as np


def _tuvel_ve_cevirme(H, sol_yukseklik, sol_genislik, sag_yukseklik, sag_genislik, pad=2):
    """Warped sol + sağ (sabit) köşelerine göre tuval boyutu ve öteleme matrisi T."""
    sol_kose = np.float32(
        [[0, 0], [sol_genislik, 0], [sol_genislik, sol_yukseklik], [0, sol_yukseklik]]
    ).reshape(-1, 1, 2)
    sol_donmus = cv2.perspectiveTransform(sol_kose, H).reshape(-1, 2)

    sag_kose = np.float32(
        [[0, 0], [sag_genislik, 0], [sag_genislik, sag_yukseklik], [0, sag_yukseklik]]
    )
    tum = np.vstack([sol_donmus, sag_kose])

    xmin, ymin = tum.min(axis=0)
    xmax, ymax = tum.max(axis=0)

    tx = int(np.floor(-xmin)) + pad
    ty = int(np.floor(-ymin)) + pad
    cw = int(np.ceil(xmax - xmin)) + 2 * pad
    ch = int(np.ceil(ymax - ymin)) + 2 * pad

    T = np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    return cw, ch, T


def _tuy_birlestir(warp_sol, warp_sag):
    """
    İki warped BGR görüntüyü çakışmada mesafe ağırlıklı harmanlar (seam yumuşatma).
    Her pikselde içeriğe uzaklık oranında ağırlık verilir.
    """
    g1 = cv2.cvtColor(warp_sol, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(warp_sag, cv2.COLOR_BGR2GRAY)
    m1 = (g1 > 0).astype(np.uint8) * 255
    m2 = (g2 > 0).astype(np.uint8) * 255

    d1 = cv2.distanceTransform(m1, cv2.DIST_L2, 5).astype(np.float32)
    d2 = cv2.distanceTransform(m2, cv2.DIST_L2, 5).astype(np.float32)
    payda = d1 + d2 + 1e-6
    w1 = (d1 / payda)[..., None]
    w2 = (d2 / payda)[..., None]

    sol_f = warp_sol.astype(np.float32)
    sag_f = warp_sag.astype(np.float32)
    sonuc = sol_f * w1 + sag_f * w2

    sade_sol = (m1 > 0) & (m2 == 0)
    sade_sag = (m2 > 0) & (m1 == 0)
    sonuc[sade_sol] = sol_f[sade_sol]
    sonuc[sade_sag] = sag_f[sade_sag]

    return np.clip(sonuc, 0, 255).astype(np.uint8)


def panorama_birlestir(sol_bgr, sag_bgr, H):
    """
    5. Aşama: Sol görüntüyü H ile sağ düzleme bük (warp), tek tuvalde sağ ile birleştir,
    çakışmada tüy harmanlama (feather) ile seam yumuşat.

    H: sol piksel koordinatlarını sağ görüntü düzlemine taşıyan homografi
       (homografi.donusum_matrisi_hesapla ile aynı yön).
    """
    if H is None:
        print("HATA: Homografi yok, birleştirme atlandı.")
        return None

    hL, wL = sol_bgr.shape[:2]
    hR, wR = sag_bgr.shape[:2]

    cw, ch, T = _tuvel_ve_cevirme(H, hL, wL, hR, wR)
    M_sol = T @ H
    M_sag = T

    warp_sol = cv2.warpPerspective(sol_bgr, M_sol, (cw, ch), flags=cv2.INTER_LINEAR)
    warp_sag = cv2.warpPerspective(sag_bgr, M_sag, (cw, ch), flags=cv2.INTER_LINEAR)

    panorama = _tuy_birlestir(warp_sol, warp_sag)
    return panorama
