"""Các hàm GLM dùng cho định phí: ma trận thiết kế, ước lượng, sai số chuẩn, kiểm định, chọn biến.

Mô hình tần suất : Poisson, hàm liên kết log, biến mục tiêu = số vụ / exposure, trọng số = exposure
                   (tương đương Poisson trên số vụ với offset log(exposure)).
Mô hình mức độ   : Gamma, hàm liên kết log, biến mục tiêu = tiền bồi thường TB mỗi vụ, trọng số = số vụ.
"""
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance

from . import cau_hinh as ch


class ThietKe:
    """Mã hoá one-hot các yếu tố, bỏ cột của nhóm cơ sở (hệ số của nhóm cơ sở = 1)."""

    def __init__(self, yeu_to: list, cac_nhom: dict, nhom_co_so: dict):
        self.yeu_to = list(yeu_to)
        self.cac_nhom = cac_nhom
        self.nhom_co_so = nhom_co_so
        self.cot = [(y, g) for y in self.yeu_to for g in cac_nhom[y] if g != nhom_co_so[y]]
        self._chi_so = {c: i for i, c in enumerate(self.cot)}

    def bien_doi(self, df: pd.DataFrame) -> sp.csr_matrix:
        dong, cot = [], []
        for y in self.yeu_to:
            nhom = self.cac_nhom[y]
            ma = pd.Categorical(df[ch.cot_nhom(y)], categories=nhom).codes
            if (ma < 0).any():
                raise ValueError(f"Có giá trị không thuộc nhóm nào ở yếu tố {y}")
            anh_xa = np.array([self._chi_so.get((y, g), -1) for g in nhom])
            c = anh_xa[ma]
            co = c >= 0
            dong.append(np.flatnonzero(co))
            cot.append(c[co])
        dong = np.concatenate(dong) if dong else np.array([], dtype=int)
        cot = np.concatenate(cot) if cot else np.array([], dtype=int)
        return sp.csr_matrix((np.ones(len(dong)), (dong, cot)), shape=(len(df), len(self.cot)))


class _HangSo:
    """Mô hình không có yếu tố nào (chỉ hệ số chặn)."""

    def __init__(self, gia_tri):
        self.intercept_ = float(np.log(gia_tri))
        self.coef_ = np.array([])

    def predict(self, X):
        return np.full(X.shape[0], np.exp(self.intercept_))


def khop_glm(X, y, w, phan_phoi: str):
    """Ước lượng hợp lý cực đại (không phạt) bằng scikit-learn."""
    if X.shape[1] == 0:
        return _HangSo(np.average(y, weights=w))
    lop = PoissonRegressor if phan_phoi == "poisson" else GammaRegressor
    mo_hinh = lop(alpha=0.0, solver="newton-cholesky", max_iter=1000, tol=1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mo_hinh.fit(X, y, sample_weight=w)
    return mo_hinh


def do_lech(y, mu, w, phan_phoi: str) -> float:
    """Tổng deviance có trọng số."""
    ham = mean_poisson_deviance if phan_phoi == "poisson" else mean_gamma_deviance
    return float(ham(y, mu, sample_weight=w) * np.sum(w))


def he_so_phan_tan(y, mu, w, so_tham_so: int, phan_phoi: str) -> float:
    """Poisson: cố định 1. Gamma: ước lượng Pearson."""
    if phan_phoi == "poisson":
        return 1.0
    return float(np.sum(w * (y - mu) ** 2 / mu ** 2) / (len(y) - so_tham_so))


def sai_so_chuan(mo_hinh, X, y, w, phan_phoi: str):
    """Sai số chuẩn Wald từ ma trận thông tin Fisher. Phần tử đầu tiên là của hệ số chặn."""
    mu = mo_hinh.predict(X)
    Xi = sp.hstack([sp.csr_matrix(np.ones((X.shape[0], 1))), X]).tocsr()
    phi = he_so_phan_tan(y, mu, w, Xi.shape[1], phan_phoi)
    trong_so = w * mu if phan_phoi == "poisson" else w
    thong_tin = (Xi.T @ sp.diags(trong_so) @ Xi).toarray() / phi
    hiep_phuong_sai = np.linalg.inv(thong_tin)
    return np.sqrt(np.diag(hiep_phuong_sai)), phi


def kiem_dinh_tung_yeu_to(df, y, w, yeu_to: list, cac_nhom, nhom_co_so, phan_phoi: str):
    """Kiểm định bỏ-từng-yếu-tố (likelihood ratio): Poisson dùng χ², Gamma dùng F (có tham số phân tán)."""
    tk_day_du = ThietKe(yeu_to, cac_nhom, nhom_co_so)
    X = tk_day_du.bien_doi(df)
    mh = khop_glm(X, y, w, phan_phoi)
    mu = mh.predict(X)
    d_day_du = do_lech(y, mu, w, phan_phoi)
    so_tham_so = X.shape[1] + 1
    phi = he_so_phan_tan(y, mu, w, so_tham_so, phan_phoi)
    ket_qua = []
    for yt in yeu_to:
        con_lai = [v for v in yeu_to if v != yt]
        tk = ThietKe(con_lai, cac_nhom, nhom_co_so)
        X_rut = tk.bien_doi(df)
        mu_rut = khop_glm(X_rut, y, w, phan_phoi).predict(X_rut)
        chenh = do_lech(y, mu_rut, w, phan_phoi) - d_day_du
        bac_tu_do = len(cac_nhom[yt]) - 1
        if phan_phoi == "poisson":
            thong_ke = chenh
            p = stats.chi2.sf(chenh, bac_tu_do)
        else:
            thong_ke = chenh / bac_tu_do / phi
            p = stats.f.sf(thong_ke, bac_tu_do, len(y) - so_tham_so)
        ket_qua.append({"yeu_to": yt, "bac_tu_do": bac_tu_do, "thong_ke": float(thong_ke),
                        "p_value": float(p)})
    return ket_qua


def deviance_kiem_dinh_cheo(df, y, w, yeu_to: list, cac_nhom, nhom_co_so, phan_phoi: str,
                            so_phan=ch.SO_PHAN_CV, so_lan_lap=ch.SO_LAN_LAP_CV, hat=ch.HAT_GIONG):
    """Deviance trung bình trên mỗi phần giữ lại khi kiểm định chéo `so_phan` phần, lặp `so_lan_lap` lần.

    Cách chia cố định theo hạt giống nên mọi tập yếu tố được so trên cùng các phần dữ liệu.
    """
    X = ThietKe(yeu_to, cac_nhom, nhom_co_so).bien_doi(df)
    ket_qua = []
    for lan in range(so_lan_lap):
        phan = np.random.default_rng(hat + lan).integers(0, so_phan, len(y))
        for k in range(so_phan):
            hl, kt = np.flatnonzero(phan != k), np.flatnonzero(phan == k)
            mo_hinh = khop_glm(X[hl], y[hl], w[hl], phan_phoi)
            ket_qua.append(do_lech(y[kt], mo_hinh.predict(X[kt]), w[kt], phan_phoi) / np.sum(w[kt]))
    return np.array(ket_qua)


def chon_bien_lui_dan(df, y, w, yeu_to: list, cac_nhom, nhom_co_so, phan_phoi: str,
                      muc_y_nghia=ch.MUC_Y_NGHIA, ghi=print):
    """Loại lùi dần. Một yếu tố được giữ khi thoả cả hai điều kiện:
       (1) có ý nghĩa thống kê: p-value của kiểm định LR <= mức ý nghĩa;
       (2) giúp dự báo tốt hơn rõ rệt: bỏ yếu tố đi làm deviance kiểm định chéo tăng hơn
           1 sai số chuẩn (quy tắc 1-SE, ưu tiên mô hình đơn giản khi khác biệt nằm trong nhiễu).
    Mỗi vòng: nếu có yếu tố vi phạm (1) thì bỏ yếu tố có p-value lớn nhất; nếu không thì bỏ yếu tố
    có tỷ số (mức tăng deviance / sai số chuẩn) nhỏ nhất khi tỷ số này <= 1; dừng khi tất cả thoả.
    """
    con_lai = list(yeu_to)
    da_loai = []
    while con_lai:
        kq = kiem_dinh_tung_yeu_to(df, y, w, con_lai, cac_nhom, nhom_co_so, phan_phoi)
        khong_y_nghia = [r for r in kq if r["p_value"] > muc_y_nghia]
        if khong_y_nghia:
            bo = max(khong_y_nghia, key=lambda r: r["p_value"])
            bo.update(ly_do="p_value", cv_thay_doi=None, cv_sai_so=None, cv_ty_so=None)
            ghi(f"    loại {bo['yeu_to']}: không có ý nghĩa thống kê (p = {bo['p_value']:.3f})")
        else:
            cv_hien_tai = deviance_kiem_dinh_cheo(df, y, w, con_lai, cac_nhom, nhom_co_so, phan_phoi)
            for r in kq:
                cv_khi_bo = deviance_kiem_dinh_cheo(df, y, w, [v for v in con_lai if v != r["yeu_to"]],
                                                    cac_nhom, nhom_co_so, phan_phoi)
                chenh = (cv_khi_bo - cv_hien_tai) / cv_hien_tai.mean()     # > 0: yếu tố có ích
                r["cv_thay_doi"] = float(chenh.mean())
                r["cv_sai_so"] = float(chenh.std(ddof=1) / np.sqrt(len(chenh)))
                r["cv_ty_so"] = r["cv_thay_doi"] / r["cv_sai_so"]
            bo = min(kq, key=lambda r: r["cv_ty_so"])
            if bo["cv_ty_so"] > 1:
                ghi(f"    giữ {len(con_lai)} yếu tố: {', '.join(con_lai)}")
                return con_lai, kq, da_loai
            bo["ly_do"] = "kiem_dinh_cheo"
            ghi(f"    loại {bo['yeu_to']}: có ý nghĩa (p = {bo['p_value']:.4f}) nhưng chỉ giảm deviance kiểm định "
                f"chéo {bo['cv_thay_doi'] * 100:+.3f}% (sai số chuẩn {bo['cv_sai_so'] * 100:.3f}%)")
        da_loai.append(bo)
        con_lai.remove(bo["yeu_to"])
    return con_lai, [], da_loai


def bang_he_so(mo_hinh, thiet_ke: ThietKe, sai_so, tat_ca_yeu_to: list):
    """Hệ số nhân (exp(beta)) và khoảng tin cậy 95% cho mọi nhóm; yếu tố không có trong mô hình = 1."""
    beta = dict(zip(thiet_ke.cot, mo_hinh.coef_))
    se = dict(zip(thiet_ke.cot, sai_so[1:]))
    ket_qua = {}
    for y in tat_ca_yeu_to:
        ket_qua[y] = {}
        for g in thiet_ke.cac_nhom[y]:
            b, s = beta.get((y, g), 0.0), se.get((y, g), 0.0)
            ket_qua[y][g] = {"he_so": float(np.exp(b)),
                             "ktc": [float(np.exp(b - 1.96 * s)), float(np.exp(b + 1.96 * s))]}
    return ket_qua
