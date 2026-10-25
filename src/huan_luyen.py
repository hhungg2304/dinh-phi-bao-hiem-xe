"""Huấn luyện mô hình định phí và xuất kết quả cho web.

Chạy từ thư mục gốc của dự án:   python -m src.huan_luyen

Đầu vào : data/freMTPL2freq.csv, data/freMTPL2sev.csv
Đầu ra  : models/bieu_phi.json  – phí cơ sở và hệ số từng nhóm (web dùng để tính phí)
          models/thong_ke.json  – số liệu tổng hợp, kiểm định, đánh giá mô hình (web dùng để vẽ biểu đồ)
"""
import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from . import cau_hinh as ch
from .bieu_phi import tinh_phi
from .dinh_dang import he_so, phan_tram, so, tien
from .mo_hinh import ThietKe, bang_he_so, chon_bien_lui_dan, do_lech, khop_glm, sai_so_chuan
from .xu_ly_du_lieu import doc_du_lieu, lam_sach

COT_SO_HGB = ["DrivAge", "VehAge", "VehPower", "BonusMalus", "Density"]
COT_DM_HGB = ["VehBrand", "VehGas", "Region"]

# Ba hồ sơ minh hoạ cho báo cáo
HO_SO_MAU = [
    {"ten": "Người lái trẻ, mới tham gia, sống ở thành phố",
     "ho_so": {"DrivAge": 20, "BonusMalus": 100, "VehAge": 2, "VehPower": 6, "VehBrand": "B12",
               "VehGas": "Regular", "Density": 3000, "Region": "Ile-de-France"}},
    {"ten": "Người lái trung niên, nhiều năm không tai nạn, sống ở nông thôn",
     "ho_so": {"DrivAge": 45, "BonusMalus": 50, "VehAge": 8, "VehPower": 6, "VehBrand": "B1",
               "VehGas": "Diesel", "Density": 40, "Region": "Bretagne"}},
    {"ten": "Người lái lớn tuổi, sống ở đô thị đông dân",
     "ho_so": {"DrivAge": 72, "BonusMalus": 50, "VehAge": 12, "VehPower": 5, "VehBrand": "B2",
               "VehGas": "Regular", "Density": 12000, "Region": "Provence-Alpes-Cotes-D'Azur"}},
]


def ghi(thong_diep: str):
    print(thong_diep, flush=True)


def _sach(x):
    """Số Python thuần cho JSON; NaN/inf -> None."""
    if x is None:
        return None
    x = float(x)
    return None if (math.isnan(x) or math.isinf(x)) else x


def luu_json(du_lieu, duong_dan):
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    duong_dan.write_text(json.dumps(du_lieu, ensure_ascii=False, indent=1), encoding="utf-8")


def chia_tap(df):
    ngau_nhien = np.random.default_rng(ch.HAT_GIONG).random(len(df))
    kiem_tra = ngau_nhien < ch.TY_LE_KIEM_TRA
    return df.loc[~kiem_tra].reset_index(drop=True), df.loc[kiem_tra].reset_index(drop=True)


def muc_tieu_tan_suat(df):
    """y = số vụ / năm, trọng số = exposure."""
    w = df["Exposure"].to_numpy(float)
    return df["SoVuMH"].to_numpy(float) / w, w


def muc_tieu_muc_do(df_co_vu):
    """y = tiền bồi thường (đã chặn) trung bình mỗi vụ, trọng số = số vụ."""
    w = df_co_vu["SoVu"].to_numpy(float)
    return df_co_vu["TienBTChan"].to_numpy(float) / w, w


def dac_trung_hgb(df, cac_nhom):
    so_lieu = df[COT_SO_HGB].to_numpy(float)
    danh_muc = np.column_stack([pd.Categorical(df[c], categories=cac_nhom[c]).codes for c in COT_DM_HGB])
    return np.column_stack([so_lieu, danh_muc]), [False] * len(COT_SO_HGB) + [True] * len(COT_DM_HGB)


def lorenz(chi_phi, du_bao, exposure, so_diem=201):
    """Đường Lorenz: sắp hợp đồng từ rủi ro thấp đến cao theo dự báo; Gini = 1 − 2 × diện tích dưới đường."""
    thu_tu = np.argsort(du_bao, kind="mergesort")
    e, c = exposure[thu_tu], chi_phi[thu_tu]
    x = np.r_[0.0, np.cumsum(e)] / e.sum()
    y = np.r_[0.0, np.cumsum(c)] / c.sum()
    dien_tich = np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2)
    x_deu = np.linspace(0, 1, so_diem)
    return float(1 - 2 * dien_tich), np.interp(x_deu, x, y)


def thap_phan_vi(du_bao, chi_phi, exposure, so_nhom=10):
    """Chia 10 nhóm bằng nhau về exposure theo phí dự báo; so phí dự báo và chi phí thực tế mỗi nhóm."""
    thu_tu = np.argsort(du_bao, kind="mergesort")
    e, p, c = exposure[thu_tu], du_bao[thu_tu], chi_phi[thu_tu]
    vi_tri = (np.cumsum(e) - e / 2) / e.sum()
    nhom = np.minimum((vi_tri * so_nhom).astype(int), so_nhom - 1)
    ket_qua = []
    for k in range(so_nhom):
        m = nhom == k
        ket_qua.append({"nhom": k + 1, "exposure": float(e[m].sum()),
                        "du_bao": float((p[m] * e[m]).sum() / e[m].sum()),
                        "thuc_te": float(c[m].sum() / e[m].sum())})
    return ket_qua


def thong_ke_yeu_to(df, yt, nhom, pred_ts, pred_md, hs_ts, hs_md):
    """Số liệu thực tế theo nhóm + 'ảnh hưởng riêng' của yếu tố theo GLM.

    Ảnh hưởng riêng: tần suất (mức bồi thường) trung bình của danh mục nếu mọi hợp đồng cùng thuộc
    nhóm g, các yếu tố khác giữ nguyên như thực tế. Bằng A × hệ số(g), trong đó A là mức trung bình
    sau khi trung hoà yếu tố đang xét.
    """
    ma = pd.Categorical(df[ch.cot_nhom(yt)], categories=nhom).codes
    k = len(nhom)

    def cong(gia_tri):
        return np.bincount(ma, weights=gia_tri, minlength=k)

    E = df["Exposure"].to_numpy(float)
    so_hd = np.bincount(ma, minlength=k)
    expo, vu_mh, vu = cong(E), cong(df["SoVuMH"].to_numpy(float)), cong(df["SoVu"].to_numpy(float))
    tien_chan, tien_goc = cong(df["TienBTChan"].to_numpy(float)), cong(df["TienBT"].to_numpy(float))

    r_ts = np.array([hs_ts[yt][g]["he_so"] for g in nhom])
    r_md = np.array([hs_md[yt][g]["he_so"] for g in nhom])
    a_ts = np.sum(E * pred_ts / r_ts[ma]) / E.sum()
    vu_ky_vong = E * pred_ts
    a_md = np.sum(vu_ky_vong * pred_md / r_md[ma]) / vu_ky_vong.sum()

    hang = []
    for j, g in enumerate(nhom):
        hang.append({
            "nhom": g,
            "so_hop_dong": int(so_hd[j]),
            "exposure": _sach(expo[j]),
            "ty_trong_exposure": _sach(expo[j] / expo.sum()),
            "so_vu": _sach(vu[j]),
            "ty_trong_vu": _sach(vu[j] / vu.sum()),
            "tan_suat": _sach(vu_mh[j] / expo[j]) if expo[j] > 0 else None,
            "tan_suat_rieng": _sach(a_ts * r_ts[j]),
            "muc_bt": _sach(tien_chan[j] / vu[j]) if vu[j] > 0 else None,
            "muc_bt_rieng": _sach(a_md * r_md[j]),
            "tien_bt": _sach(tien_goc[j]),
            "phi_thuan_thuc_te": _sach(tien_goc[j] / expo[j]) if expo[j] > 0 else None,
        })
    return hang


def main():
    bat_dau = time.time()

    # ------------------------------------------------------------------ 1. Dữ liệu
    ghi("1/6 Đọc và làm sạch dữ liệu")
    tan_suat_goc, boi_thuong_goc = doc_du_lieu()
    df, vu, cac_nhom, nhat_ky = lam_sach(tan_suat_goc, boi_thuong_goc)
    for buoc in nhat_ky:
        ghi(f"   - {buoc['buoc']}: {buoc['chi_tiet']}")

    # Nhóm cơ sở của mỗi yếu tố = nhóm có tổng exposure lớn nhất
    nhom_co_so = {}
    for yt in ch.THU_TU_YEU_TO:
        expo = df.groupby(ch.cot_nhom(yt), observed=False)["Exposure"].sum()
        nhom_co_so[yt] = str(expo.idxmax())
    ghi("   Nhóm cơ sở: " + ", ".join(f"{yt}={g}" for yt, g in nhom_co_so.items()))

    train, test = chia_tap(df)
    train_vu, test_vu = train[train["SoVu"] > 0], test[test["SoVu"] > 0]
    ghi(f"   Huấn luyện: {so(len(train))} hợp đồng, kiểm tra: {so(len(test))} hợp đồng")

    # ------------------------------------------------------------------ 2. Chọn yếu tố
    ghi("2/6 Chọn yếu tố bằng loại lùi dần trên tập huấn luyện "
        "(kiểm định likelihood ratio mức 5% + kiểm định chéo 5 phần × 4 lần, quy tắc 1-SE)")
    y_tr, w_tr = muc_tieu_tan_suat(train)
    ys_tr, ws_tr = muc_tieu_muc_do(train_vu)
    ghi("   Mô hình tần suất (Poisson):")
    yt_ts, kd_ts, loai_ts = chon_bien_lui_dan(train, y_tr, w_tr, ch.THU_TU_YEU_TO, cac_nhom, nhom_co_so,
                                              "poisson", ghi=ghi)
    ghi("   Mô hình mức độ (Gamma):")
    yt_md, kd_md, loai_md = chon_bien_lui_dan(train_vu, ys_tr, ws_tr, ch.THU_TU_YEU_TO, cac_nhom, nhom_co_so,
                                              "gamma", ghi=ghi)

    def bang_kiem_dinh(giu, kq, loai):
        tra = {r["yeu_to"]: {**r, "giu": True} for r in kq}
        tra.update({r["yeu_to"]: {**r, "giu": False} for r in loai})
        return tra

    kiem_dinh = {"tan_suat": bang_kiem_dinh(yt_ts, kd_ts, loai_ts),
                 "muc_do": bang_kiem_dinh(yt_md, kd_md, loai_md)}

    # ------------------------------------------------------------------ 3. Đánh giá trên tập kiểm tra
    ghi("3/6 Đánh giá trên tập kiểm tra: GLM so với HistGradientBoosting")
    tk_ts, tk_md = ThietKe(yt_ts, cac_nhom, nhom_co_so), ThietKe(yt_md, cac_nhom, nhom_co_so)
    glm_ts_tr = khop_glm(tk_ts.bien_doi(train), y_tr, w_tr, "poisson")
    glm_md_tr = khop_glm(tk_md.bien_doi(train_vu), ys_tr, ws_tr, "gamma")

    X_hgb_tr, la_danh_muc = dac_trung_hgb(train, cac_nhom)
    hgb_ts = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=1000, max_leaf_nodes=31, min_samples_leaf=1000,
        l2_regularization=1.0, categorical_features=la_danh_muc, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=50, random_state=ch.HAT_GIONG)
    hgb_ts.fit(X_hgb_tr, y_tr, sample_weight=w_tr)
    X_hgb_tr_vu, _ = dac_trung_hgb(train_vu, cac_nhom)
    hgb_md = HistGradientBoostingRegressor(
        loss="gamma", learning_rate=0.03, max_iter=1000, max_leaf_nodes=8, min_samples_leaf=300,
        l2_regularization=1.0, categorical_features=la_danh_muc, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=50, random_state=ch.HAT_GIONG)
    hgb_md.fit(X_hgb_tr_vu, ys_tr, sample_weight=ws_tr)
    ghi(f"   HistGradientBoosting: {hgb_ts.n_iter_} vòng (tần suất), {hgb_md.n_iter_} vòng (mức độ)")

    y_te, w_te = muc_tieu_tan_suat(test)
    ys_te, ws_te = muc_tieu_muc_do(test_vu)
    X_hgb_te, _ = dac_trung_hgb(test, cac_nhom)
    X_hgb_te_vu, _ = dac_trung_hgb(test_vu, cac_nhom)

    du_bao_ts = {"GLM Poisson": glm_ts_tr.predict(tk_ts.bien_doi(test)),
                 "HistGradientBoosting": hgb_ts.predict(X_hgb_te)}
    du_bao_md_vu = {"GLM Gamma": glm_md_tr.predict(tk_md.bien_doi(test_vu)),
                    "HistGradientBoosting": hgb_md.predict(X_hgb_te_vu)}
    du_bao_md = {"GLM": glm_md_tr.predict(tk_md.bien_doi(test)),
                 "HistGradientBoosting": hgb_md.predict(X_hgb_te)}

    d_rong_ts = do_lech(y_te, np.full_like(y_te, np.average(y_tr, weights=w_tr)), w_te, "poisson")
    d_rong_md = do_lech(ys_te, np.full_like(ys_te, np.average(ys_tr, weights=ws_tr)), ws_te, "gamma")
    danh_gia_ts = [{"mo_hinh": "Không dùng yếu tố nào", "deviance_tb": d_rong_ts / w_te.sum(), "d2": 0.0}]
    for ten, mu in du_bao_ts.items():
        d = do_lech(y_te, mu, w_te, "poisson")
        danh_gia_ts.append({"mo_hinh": ten, "deviance_tb": d / w_te.sum(), "d2": 1 - d / d_rong_ts})
    danh_gia_md = [{"mo_hinh": "Không dùng yếu tố nào", "deviance_tb": d_rong_md / ws_te.sum(), "d2": 0.0}]
    for ten, mu in du_bao_md_vu.items():
        d = do_lech(ys_te, mu, ws_te, "gamma")
        danh_gia_md.append({"mo_hinh": ten, "deviance_tb": d / ws_te.sum(), "d2": 1 - d / d_rong_md})

    chi_phi_te = test["TienBTChan"].to_numpy(float)
    phi_glm = du_bao_ts["GLM Poisson"] * du_bao_md["GLM"]
    phi_hgb = du_bao_ts["HistGradientBoosting"] * du_bao_md["HistGradientBoosting"]
    gini_glm, lorenz_glm = lorenz(chi_phi_te, phi_glm, w_te)
    gini_hgb, lorenz_hgb = lorenz(chi_phi_te, phi_hgb, w_te)
    danh_gia_phi = [
        {"mo_hinh": "GLM (Poisson × Gamma)", "gini": gini_glm,
         "ty_le_du_bao_thuc_te": float((phi_glm * w_te).sum() / chi_phi_te.sum())},
        {"mo_hinh": "HistGradientBoosting", "gini": gini_hgb,
         "ty_le_du_bao_thuc_te": float((phi_hgb * w_te).sum() / chi_phi_te.sum())},
    ]
    for r in danh_gia_ts[1:] + danh_gia_md[1:]:
        ghi(f"   {r['mo_hinh']:<22} D² = {r['d2']:.4f}")
    for r in danh_gia_phi:
        ghi(f"   {r['mo_hinh']:<22} Gini = {r['gini']:.4f}, dự báo/thực tế = {r['ty_le_du_bao_thuc_te']:.3f}")

    # ------------------------------------------------------------------ 4. Mô hình cuối (toàn bộ dữ liệu)
    ghi("4/6 Ước lượng mô hình cuối trên toàn bộ dữ liệu")
    X_ts = tk_ts.bien_doi(df)
    y_all, w_all = muc_tieu_tan_suat(df)
    glm_ts = khop_glm(X_ts, y_all, w_all, "poisson")
    se_ts, _ = sai_so_chuan(glm_ts, X_ts, y_all, w_all, "poisson")

    df_vu = df[df["SoVu"] > 0]
    X_md_vu = tk_md.bien_doi(df_vu)
    ys_all, ws_all = muc_tieu_muc_do(df_vu)
    glm_md = khop_glm(X_md_vu, ys_all, ws_all, "gamma")
    se_md, phi_md = sai_so_chuan(glm_md, X_md_vu, ys_all, ws_all, "gamma")

    hs_ts = bang_he_so(glm_ts, tk_ts, se_ts, ch.THU_TU_YEU_TO)
    hs_md = bang_he_so(glm_md, tk_md, se_md, ch.THU_TU_YEU_TO)

    pred_ts = glm_ts.predict(X_ts)
    pred_md = glm_md.predict(tk_md.bien_doi(df))
    E = df["Exposure"].to_numpy(float)
    k_can_bang = df["TienBTChan"].sum() / np.sum(E * pred_ts * pred_md)
    k_ton_that_lon = df["TienBT"].sum() / df["TienBTChan"].sum()
    k = k_can_bang * k_ton_that_lon

    tan_suat_co_so = float(np.exp(glm_ts.intercept_))
    muc_bt_chan_co_so = float(np.exp(glm_md.intercept_))
    ghi(f"   Tần suất cơ sở {phan_tram(tan_suat_co_so, 2)}, mức bồi thường cơ sở (đã chặn) {tien(muc_bt_chan_co_so)}")
    ghi(f"   Hệ số cân bằng {so(k_can_bang, 4)} × hệ số tổn thất lớn {so(k_ton_that_lon, 4)} = {so(k, 4)}")
    ghi(f"   Tham số phân tán Gamma φ = {so(phi_md, 3)}")

    # ------------------------------------------------------------------ 5. Biểu phí + số liệu
    ghi("5/6 Tạo biểu phí và số liệu tổng hợp")
    bieu_phi = {
        "mo_ta": "Biểu phí bảo hiểm TNDS xe cơ giới – GLM Poisson (tần suất) × Gamma (mức độ), dữ liệu freMTPL2",
        "don_vi_tien": "EUR",
        "nguong_ton_that_lon": ch.NGUONG_TON_THAT_LON,
        "phi_co_so": {
            "tan_suat": tan_suat_co_so,
            "muc_bt_chan": muc_bt_chan_co_so,
            "he_so_can_bang": float(k_can_bang),
            "he_so_ton_that_lon": float(k_ton_that_lon),
            "he_so_dieu_chinh": float(k),
            "muc_bt": muc_bt_chan_co_so * float(k),
            "phi_thuan": tan_suat_co_so * muc_bt_chan_co_so * float(k),
        },
        "thu_tu_yeu_to": ch.THU_TU_YEU_TO,
        "yeu_to": {},
    }
    for yt in ch.THU_TU_YEU_TO:
        cau_hinh = ch.YEU_TO[yt]
        nhom = cac_nhom[yt]
        muc = {
            "ten": cau_hinh["ten"],
            "kieu": cau_hinh["kieu"],
            "don_vi": cau_hinh.get("don_vi", ""),
            "nhom": nhom,
            "nhan": {g: cau_hinh.get("nhan", {}).get(g, g) for g in nhom},
            "co_so": nhom_co_so[yt],
            "trong_mo_hinh": {"tan_suat": yt in yt_ts, "muc_do": yt in yt_md},
            "he_so_tan_suat": {g: hs_ts[yt][g]["he_so"] for g in nhom},
            "ktc_tan_suat": {g: hs_ts[yt][g]["ktc"] for g in nhom},
            "he_so_muc_do": {g: hs_md[yt][g]["he_so"] for g in nhom},
            "ktc_muc_do": {g: hs_md[yt][g]["ktc"] for g in nhom},
            "he_so_phi": {g: hs_ts[yt][g]["he_so"] * hs_md[yt][g]["he_so"] for g in nhom},
        }
        if cau_hinh["kieu"] == "so":
            muc["can_tren"] = [t for _, t in cau_hinh["nhom"]]
            muc["gia_tri_nho_nhat"] = int(df[yt].min())
            muc["gia_tri_lon_nhat"] = int(df[yt].max())
        bieu_phi["yeu_to"][yt] = muc

    # Kiểm tra: phí tính bằng biểu phí (Python thuần) phải khớp dự báo của mô hình
    mau = np.random.default_rng(0).choice(len(df), size=3000, replace=False)
    lech = 0.0
    for i in mau:
        hang = df.iloc[i]
        kq = tinh_phi(bieu_phi, {yt: hang[yt] for yt in ch.THU_TU_YEU_TO})
        lech = max(lech, abs(kq["phi_thuan"] / (pred_ts[i] * pred_md[i] * k) - 1))
    if lech > 1e-9:
        raise AssertionError(f"Biểu phí không khớp mô hình (sai lệch {lech:.2e})")
    ghi(f"   Kiểm tra biểu phí: khớp dự báo GLM trên 3.000 hợp đồng (sai lệch lớn nhất {lech:.1e})")

    tong_e = df["Exposure"].sum()
    danh_muc = {
        "so_hop_dong": int(len(df)),
        "exposure": float(tong_e),
        "so_vu": int(df["SoVu"].sum()),
        "tan_suat": float(df["SoVuMH"].sum() / tong_e),
        "muc_bt": float(df["TienBT"].sum() / df["SoVu"].sum()),
        "muc_bt_chan": float(df["TienBTChan"].sum() / df["SoVu"].sum()),
        "phi_thuan": float(df["TienBT"].sum() / tong_e),
        "tong_tien_bt": float(df["TienBT"].sum()),
    }
    bieu_phi["danh_muc"] = danh_muc

    theo_yeu_to = {yt: thong_ke_yeu_to(df, yt, cac_nhom[yt], pred_ts, pred_md, hs_ts, hs_md)
                   for yt in ch.THU_TU_YEU_TO}

    so_tien = vu["ClaimAmount"].to_numpy(float)
    lg = np.log10(so_tien)
    moc = np.round(np.arange(0.0, math.ceil(lg.max() * 10) / 10 + 0.1, 0.1), 1)
    dem, _ = np.histogram(lg, bins=moc)
    phan_phoi = {
        "moc_log10": moc.tolist(), "so_vu": dem.astype(int).tolist(),
        "trung_vi": float(np.median(so_tien)), "trung_binh": float(so_tien.mean()),
        "lon_nhat": float(so_tien.max()), "nguong": ch.NGUONG_TON_THAT_LON,
        "so_vu_vuot_nguong": int((so_tien > ch.NGUONG_TON_THAT_LON).sum()),
        "ty_trong_tien_vuot_nguong": float((so_tien - np.minimum(so_tien, ch.NGUONG_TON_THAT_LON)).sum() / so_tien.sum()),
        "ty_trong_vu_muc_khoan": float(((so_tien >= 1100) & (so_tien <= 1250)).mean()),
    }

    ho_so_mau = []
    for mau_hs in HO_SO_MAU:
        kq = tinh_phi(bieu_phi, mau_hs["ho_so"], 0.20, 0.05)
        ho_so_mau.append({"ten": mau_hs["ten"], "ho_so": mau_hs["ho_so"],
                          "tan_suat": kq["tan_suat"], "muc_bt": kq["muc_bt"],
                          "phi_thuan": kq["phi_thuan"], "phi_ban": kq["phi_ban"]})
        ghi(f"   {mau_hs['ten']}: phí thuần {tien(kq['phi_thuan'])}/năm "
            f"({he_so(kq['phi_thuan'] / danh_muc['phi_thuan'])} trung bình)")

    thong_ke = {
        "ngay_tao": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "phien_ban_sklearn": sklearn.__version__,
        "nhat_ky_lam_sach": nhat_ky,
        "danh_muc": danh_muc,
        "theo_yeu_to": theo_yeu_to,
        "phan_phoi_boi_thuong": phan_phoi,
        "kiem_dinh": kiem_dinh,
        "phi_md": phi_md,
        "danh_gia": {
            "so_hop_dong_huan_luyen": int(len(train)),
            "so_hop_dong_kiem_tra": int(len(test)),
            "tan_suat": danh_gia_ts,
            "muc_do": danh_gia_md,
            "phi_thuan": danh_gia_phi,
            "lorenz": {"x": np.linspace(0, 1, len(lorenz_glm)).tolist(),
                       "GLM": lorenz_glm.tolist(), "HistGradientBoosting": lorenz_hgb.tolist()},
            "thap_phan_vi": thap_phan_vi(phi_glm, chi_phi_te, w_te),
        },
        "ho_so_mau": ho_so_mau,
    }

    # ------------------------------------------------------------------ 6. Lưu
    ghi("6/6 Lưu kết quả")
    luu_json(bieu_phi, ch.FILE_BIEU_PHI)
    luu_json(thong_ke, ch.FILE_THONG_KE)
    ghi(f"   {ch.FILE_BIEU_PHI.relative_to(ch.THU_MUC_GOC)} và {ch.FILE_THONG_KE.relative_to(ch.THU_MUC_GOC)}")
    ghi(f"Xong sau {time.time() - bat_dau:.0f} giây.")


if __name__ == "__main__":
    main()
