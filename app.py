"""Web định phí bảo hiểm TNDS xe cơ giới – một trang: tính phí + phân tích dữ liệu + chi tiết mô hình.

Chạy:  streamlit run app.py
Web chỉ đọc kết quả đã huấn luyện trong models/ (không cần dữ liệu gốc, không cần scikit-learn).
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src import bieu_do
from src.bieu_phi import tinh_phi
from src.dinh_dang import p_value, phan_tram, so, tien

st.set_page_config(page_title="Định phí bảo hiểm xe cơ giới", page_icon="🚗", layout="wide")

THU_MUC_MO_HINH = Path(__file__).resolve().parent / "models"
TIEU_DE = {
    "DrivAge": "Tuổi người lái",
    "BonusMalus": "Hệ số BonusMalus",
    "VehAge": "Tuổi xe (năm)",
    "VehPower": "Công suất xe",
    "VehBrand": "Hãng xe (mã ẩn danh)",
    "VehGas": "Nhiên liệu",
    "Density": "Mật độ dân số (người/km²)",
    "Region": "Vùng",
}
TEN_NGAN = {"DrivAge": "tuổi lái", "BonusMalus": "BonusMalus", "VehAge": "tuổi xe", "VehPower": "công suất",
            "VehBrand": "hãng", "VehGas": "nhiên liệu", "Density": "mật độ", "Region": "vùng"}
TEN_CHI_TIEU = {"tan_suat": "Tần suất bồi thường", "muc_bt": "Mức bồi thường trung bình"}


# ------------------------------------------------------------------ dữ liệu và ảnh (có bộ nhớ đệm)
@st.cache_data
def doc_ket_qua():
    duong_dan = [THU_MUC_MO_HINH / "bieu_phi.json", THU_MUC_MO_HINH / "thong_ke.json"]
    if not all(p.exists() for p in duong_dan):
        return None, None
    return tuple(json.loads(p.read_text(encoding="utf-8")) for p in duong_dan)


BIEU_PHI, THONG_KE = doc_ket_qua()
if BIEU_PHI is None:
    st.error("Chưa có kết quả mô hình trong thư mục models/. Hãy chạy `python -m src.huan_luyen` trước (xem README).")
    st.stop()
YT = BIEU_PHI["yeu_to"]
DM = BIEU_PHI["danh_muc"]
CO_SO = BIEU_PHI["phi_co_so"]


@st.cache_data(show_spinner=False)
def anh_yeu_to(yeu_to: str, chi_tieu: str, nhom_kh: str) -> bytes:
    fig = bieu_do.ve_yeu_to(THONG_KE["theo_yeu_to"][yeu_to], YT[yeu_to]["nhan"], chi_tieu, nhom_kh,
                            co_thu_tu=YT[yeu_to]["kieu"] == "so", ngang=yeu_to == "Region")
    return bieu_do.sang_png(fig)


@st.cache_data(show_spinner=False)
def anh_he_so(cac_dong: tuple) -> bytes:
    chi_tiet = [{"ten": ten, "nhan": nhan, "he_so_phi": h} for ten, nhan, h in cac_dong]
    return bieu_do.sang_png(bieu_do.ve_he_so(chi_tiet))


@st.cache_data(show_spinner=False)
def anh_co_dinh(ten: str) -> bytes:
    dg = THONG_KE["danh_gia"]
    if ten == "phan_phoi":
        fig = bieu_do.ve_phan_phoi(THONG_KE["phan_phoi_boi_thuong"])
    elif ten == "lorenz":
        fig = bieu_do.ve_lorenz(dg["lorenz"], dg["phi_thuan"])
    else:
        fig = bieu_do.ve_thap_phan_vi(dg["thap_phan_vi"])
    return bieu_do.sang_png(fig)


def mo_ta_co_so() -> str:
    """Ví dụ: tuổi lái 41–50, BonusMalus 50, tuổi xe 6–10, ..."""
    return ", ".join(f"{TEN_NGAN[y]} {YT[y]['nhan'][YT[y]['co_so']]}" for y in BIEU_PHI["thu_tu_yeu_to"])


def he_so_bonus_malus(so_nam_khong_tai_nan: int) -> int:
    """Hệ số BonusMalus theo quy tắc của Pháp: bắt đầu từ 100, mỗi năm không gây tai nạn nhân 0,95
    (làm tròn xuống), thấp nhất 50. Ví dụ: 0 năm → 100, 1 → 95, 5 → 76, 10 → 57, từ 13 năm → 50."""
    he_so = 100
    for _ in range(so_nam_khong_tai_nan):
        he_so = max(50, he_so * 95 // 100)
    return he_so


# Nhóm mật độ dân số trong biểu phí -> mô tả dễ hiểu cho người dùng
LOAI_NOI_O = {
    "≤ 50": "Nông thôn, làng xã (dưới 50 người/km²)",
    "51–100": "Thị trấn nhỏ (50–100 người/km²)",
    "101–500": "Thị trấn, ngoại ô thưa (100–500 người/km²)",
    "501–2.000": "Thành phố nhỏ, ngoại ô đông (500–2.000 người/km²)",
    "2.001–10.000": "Thành phố lớn (2.000–10.000 người/km²)",
    "> 10.000": "Trung tâm đô thị lớn như Paris (trên 10.000 người/km²)",
}


def mat_do_dai_dien(nhom: str) -> int:
    """Một giá trị mật độ thuộc nhóm đã chọn (cận trên của nhóm) để đưa vào hàm tính phí."""
    can_tren = YT["Density"]["can_tren"]
    i = YT["Density"]["nhom"].index(nhom)
    return can_tren[i] if can_tren[i] is not None else can_tren[i - 1] + 1


# ------------------------------------------------------------------ đầu trang
st.title("Định phí bảo hiểm TNDS xe cơ giới")
st.caption("Mô hình GLM: tần suất (Poisson) × mức bồi thường (Gamma), xây dựng trên bộ dữ liệu freMTPL2 – danh mục "
           "bảo hiểm trách nhiệm dân sự xe cơ giới của một công ty bảo hiểm Pháp. Số tiền tính bằng euro (€).")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Số hợp đồng", so(DM["so_hop_dong"]),
          help=f"Sau khi làm sạch; tổng cộng {so(DM['exposure'])} năm được bảo hiểm.")
c2.metric("Tần suất bồi thường", f"{phan_tram(DM['tan_suat'], 2)} / năm",
          help=f"{so(DM['so_vu'])} vụ trên {so(DM['exposure'])} năm bảo hiểm.")
c3.metric("Mức bồi thường trung bình", f"{tien(DM['muc_bt'])} / vụ", help="Tổng tiền bồi thường / số vụ.")
c4.metric("Phí thuần trung bình", f"{tien(DM['phi_thuan'])} / năm",
          help="Tổng tiền bồi thường / tổng số năm bảo hiểm: mức phí vừa đủ trả bồi thường.")

# ------------------------------------------------------------------ tính phí
st.divider()
st.header("Tính phí cho một khách hàng")
cot_nhap, cot_ket_qua = st.columns([2, 3], gap="large")

with cot_nhap:
    with st.form("form_tinh_phi"):
        st.markdown("**Người lái**")
        tuoi = st.number_input("Tuổi người lái", min_value=18, max_value=100, value=28, step=1)
        so_nam_khong_tai_nan = st.number_input(
            "Số năm có bảo hiểm liên tục mà không gây tai nạn", min_value=0, max_value=50, value=7, step=1,
            help="Mua bảo hiểm lần đầu: nhập 0. Mỗi năm không gây tai nạn có lỗi, hệ số BonusMalus giảm 5% "
                 "(bắt đầu từ 100, thấp nhất 50 sau 13 năm).")
        bonus_malus = he_so_bonus_malus(so_nam_khong_tai_nan)
        st.caption(f"→ Hệ số BonusMalus tương ứng: **{bonus_malus}** (100 khi mới mua bảo hiểm, thấp nhất 50)")
        st.markdown("**Xe**")
        a, b = st.columns(2)
        tuoi_xe = a.number_input("Tuổi xe (năm)", min_value=0, max_value=60, value=4, step=1)
        cong_suat = b.selectbox("Công suất xe", list(range(4, 16)), index=3,
                                help="Mức công suất 4–15 theo phân loại của dữ liệu gốc.")
        hang_xe = a.selectbox("Hãng xe", YT["VehBrand"]["nhom"],
                              index=YT["VehBrand"]["nhom"].index("B2"),
                              help="Dữ liệu gốc đã ẩn tên hãng, chỉ còn mã B1–B14.")
        nhien_lieu = b.radio("Nhiên liệu", YT["VehGas"]["nhom"], format_func=lambda g: YT["VehGas"]["nhan"][g],
                             index=YT["VehGas"]["nhom"].index("Diesel"), horizontal=True)
        st.markdown("**Nơi ở**")
        ds_nhom_mat_do = YT["Density"]["nhom"]
        nhom_mat_do = st.selectbox(
            "Loại nơi ở", ds_nhom_mat_do,
            index=ds_nhom_mat_do.index("501–2.000") if "501–2.000" in ds_nhom_mat_do else 0,
            format_func=lambda g: LOAI_NOI_O.get(g, f"{g} người/km²"),
            help="Khu vực nơi người lái sống; mỗi loại ứng với một khoảng mật độ dân số trong biểu phí.")
        vung = st.selectbox("Vùng", YT["Region"]["nhom"], index=YT["Region"]["nhom"].index("Ile-de-France"))
        st.markdown("**Phụ phí** (tính trên phí bán)")
        a, b = st.columns(2)
        ty_le_chi_phi = a.slider("Chi phí (%)", min_value=0, max_value=40, value=20,
                                 help="Chi phí quản lý, hoa hồng đại lý, thuế…")
        ty_le_loi_nhuan = b.slider("Lợi nhuận (%)", min_value=0, max_value=20, value=5)
        st.form_submit_button("Tính phí", type="primary")

ho_so = {"DrivAge": tuoi, "BonusMalus": bonus_malus, "VehAge": tuoi_xe, "VehPower": cong_suat,
         "VehBrand": hang_xe, "VehGas": nhien_lieu, "Density": mat_do_dai_dien(nhom_mat_do), "Region": vung}
kq = tinh_phi(BIEU_PHI, ho_so, ty_le_chi_phi / 100, ty_le_loi_nhuan / 100)
nhom_kh = {r["yeu_to"]: r["nhom"] for r in kq["chi_tiet"]}
chenh_tb = kq["phi_thuan"] / DM["phi_thuan"] - 1

with cot_ket_qua:
    m1, m2 = st.columns(2)
    m1.metric("Phí thuần", f"{tien(kq['phi_thuan'])} / năm",
              delta=f"{'+' if chenh_tb >= 0 else '-'}{so(abs(chenh_tb) * 100)}% so với trung bình danh mục",
              delta_color="inverse", help="Chi phí bồi thường kỳ vọng mỗi năm = tần suất × mức bồi thường.")
    m2.metric("Phí bán (phí toàn phần)", f"{tien(kq['phi_ban'])} / năm",
              help=f"Phí thuần / (1 − {ty_le_chi_phi}% chi phí − {ty_le_loi_nhuan}% lợi nhuận).")
    m3, m4 = st.columns(2)
    m3.metric("Tần suất dự đoán", f"{phan_tram(kq['tan_suat'], 1)} / năm",
              help=f"Trung bình khoảng 1 vụ mỗi {so(1 / kq['tan_suat'], 1)} năm.")
    m4.metric("Mức bồi thường kỳ vọng", f"{tien(kq['muc_bt'])} / vụ",
              help="Đã gồm phần tổn thất lớn (trên 100.000 €/vụ) được phân bổ đều cho mọi hợp đồng.")
    st.markdown(f"**Phí thuần = phí cơ sở {tien(CO_SO['phi_thuan'], 2)} × tích các hệ số "
                f"{so(kq['tich_he_so'], 3)} = {tien(kq['phi_thuan'], 2)} / năm**")
    st.image(anh_he_so(tuple((r["ten"], r["nhan"], r["he_so_phi"]) for r in kq["chi_tiet"])))
    st.caption(f"Hệ số của mỗi yếu tố = hệ số tần suất × hệ số mức bồi thường, so với hồ sơ cơ sở ({mo_ta_co_so()}).")
    with st.expander("Bảng hệ số của khách hàng"):
        st.dataframe(pd.DataFrame([{
            "Yếu tố": r["ten"], "Nhóm": r["nhan"], "Hệ số tần suất": so(r["he_so_tan_suat"], 3),
            "Hệ số mức bồi thường": so(r["he_so_muc_do"], 3), "Hệ số phí": so(r["he_so_phi"], 3)}
            for r in kq["chi_tiet"]]), hide_index=True)

# ------------------------------------------------------------------ phân tích dữ liệu
st.divider()
st.header("Phân tích dữ liệu")
chi_tieu = st.radio("Chỉ tiêu", list(TEN_CHI_TIEU), format_func=TEN_CHI_TIEU.get, horizontal=True,
                    label_visibility="collapsed")
st.caption("Cột xám: số liệu thực tế của từng nhóm; cột xanh: nhóm của khách hàng vừa tính phí. Điểm cam: ảnh "
           "hưởng riêng của yếu tố theo GLM, tức mức của danh mục nếu mọi hợp đồng cùng thuộc nhóm đó còn các yếu "
           "tố khác giữ nguyên. Cột cao hơn điểm cam nghĩa là một phần chênh lệch do các yếu tố khác gây ra. "
           "Khung dưới mỗi biểu đồ: tỷ trọng của nhóm trong danh mục.")

tre = THONG_KE["theo_yeu_to"]["DrivAge"][0]
GHI_CHU = {
    ("DrivAge", "tan_suat"): f"Người lái {tre['nhom']} tuổi có tần suất thực tế {phan_tram(tre['tan_suat'])}, nhưng "
                             f"ảnh hưởng riêng của tuổi chỉ {phan_tram(tre['tan_suat_rieng'])}: phần lớn chênh lệch "
                             "đến từ BonusMalus cao của người mới lái.",
    ("BonusMalus", "tan_suat"): "Yếu tố mạnh nhất: lịch sử tai nạn phản ánh trực tiếp rủi ro của người lái.",
}
LUOI = [("DrivAge", "BonusMalus"), ("VehAge", "VehPower"), ("Density", "VehBrand"), ("VehGas", "phan_phoi")]
for hang in LUOI:
    cac_cot = st.columns(2, gap="large")
    for cot, ten in zip(cac_cot, hang):
        with cot:
            if ten == "phan_phoi":
                pp = THONG_KE["phan_phoi_boi_thuong"]
                st.markdown("**Phân phối số tiền mỗi vụ bồi thường**")
                st.image(anh_co_dinh("phan_phoi"))
                st.caption(f"Phân phối lệch phải rất mạnh: một nửa số vụ dưới {tien(pp['trung_vi'])} nhưng vụ lớn nhất "
                           f"tới {tien(pp['lon_nhat'])}. Vì vậy mô hình chặn mỗi vụ ở {tien(pp['nguong'])} và phân bổ "
                           "phần vượt đều vào phí của mọi hợp đồng.")
            else:
                st.markdown(f"**{TIEU_DE[ten]}**")
                st.image(anh_yeu_to(ten, chi_tieu, nhom_kh[ten]))
                if (ten, chi_tieu) in GHI_CHU:
                    st.caption(GHI_CHU[(ten, chi_tieu)])
st.markdown(f"**{TIEU_DE['Region']}**")
st.image(anh_yeu_to("Region", chi_tieu, nhom_kh["Region"]))

with st.expander("Bảng số liệu theo nhóm"):
    yt_bang = st.selectbox("Yếu tố", BIEU_PHI["thu_tu_yeu_to"], format_func=lambda y: TIEU_DE[y])
    st.dataframe(pd.DataFrame([{
        "Nhóm": YT[yt_bang]["nhan"][h["nhom"]],
        "Số hợp đồng": so(h["so_hop_dong"]),
        "Số năm bảo hiểm": so(h["exposure"]),
        "Số vụ": so(h["so_vu"]),
        "Tần suất thực tế": phan_tram(h["tan_suat"], 2) if h["tan_suat"] is not None else "—",
        "Tần suất (ảnh hưởng riêng)": phan_tram(h["tan_suat_rieng"], 2),
        "Mức BT thực tế (đã chặn)": tien(h["muc_bt"]) if h["muc_bt"] is not None else "—",
        "Mức BT (ảnh hưởng riêng)": tien(h["muc_bt_rieng"]),
        "Phí thuần thực tế": tien(h["phi_thuan_thuc_te"]) if h["phi_thuan_thuc_te"] is not None else "—",
    } for h in THONG_KE["theo_yeu_to"][yt_bang]]), hide_index=True)

# ------------------------------------------------------------------ chi tiết mô hình
st.divider()
with st.expander("Chi tiết mô hình và phương pháp"):
    st.markdown("#### 1. Làm sạch dữ liệu")
    st.markdown("\n".join(f"- **{b['buoc']}:** {b['chi_tiet']}" for b in THONG_KE["nhat_ky_lam_sach"]))

    st.markdown("#### 2. Mô hình")
    st.markdown(
        "- **Tần suất:** GLM Poisson, hàm liên kết log. Biến mục tiêu là số vụ / năm bảo hiểm, trọng số là số năm "
        "bảo hiểm (tương đương offset log(exposure)).\n"
        f"- **Mức bồi thường:** GLM Gamma, hàm liên kết log, trên số tiền mỗi vụ đã chặn ở "
        f"{tien(BIEU_PHI['nguong_ton_that_lon'])}, trọng số là số vụ.\n"
        "- **Phí thuần** = phí cơ sở × hệ số tuổi lái × hệ số BonusMalus × … × hệ số vùng. Hệ số của một nhóm "
        "= exp(β) của nhóm đó; nhóm cơ sở (nhóm có nhiều năm bảo hiểm nhất) có hệ số 1.\n"
        f"- **Phí cơ sở** = tần suất cơ sở {phan_tram(CO_SO['tan_suat'], 2)} × mức bồi thường cơ sở "
        f"{tien(CO_SO['muc_bt_chan'])} × hệ số điều chỉnh {so(CO_SO['he_so_dieu_chinh'], 4)} = "
        f"{tien(CO_SO['phi_thuan'], 2)} / năm. Hệ số điều chỉnh = {so(CO_SO['he_so_can_bang'], 4)} (để tổng phí "
        f"khớp tổng chi phí đã chặn) × {so(CO_SO['he_so_ton_that_lon'], 4)} (phân bổ phần tổn thất vượt ngưỡng).\n"
        f"- **Hồ sơ cơ sở:** {mo_ta_co_so()}.")

    st.markdown("#### 3. Chọn yếu tố")
    st.markdown("Loại lùi dần trên tập huấn luyện (80% hợp đồng). Một yếu tố được giữ khi (1) có ý nghĩa thống kê "
                "(kiểm định likelihood ratio, p ≤ 5%) **và** (2) giúp dự báo tốt hơn rõ rệt: bỏ yếu tố đi làm "
                "deviance kiểm định chéo (5 phần × 4 lần) tăng hơn 1 sai số chuẩn.")

    def o_kiem_dinh(r):
        if r is None:
            return "—", "—", "—"
        cv = "—" if r.get("cv_thay_doi") is None else \
            f"{'+' if r['cv_thay_doi'] >= 0 else '-'}{so(abs(r['cv_thay_doi']) * 100, 3)}% ({so(r['cv_ty_so'], 1)} SE)"
        if r["giu"]:
            ket_luan = "Giữ"
        elif r.get("ly_do") == "p_value":
            ket_luan = "Loại: không có ý nghĩa"
        else:
            ket_luan = "Loại: không cải thiện dự báo"
        return p_value(r["p_value"]), cv, ket_luan

    hang_kd = []
    for y in BIEU_PHI["thu_tu_yeu_to"]:
        p1, cv1, kl1 = o_kiem_dinh(THONG_KE["kiem_dinh"]["tan_suat"].get(y))
        p2, cv2, kl2 = o_kiem_dinh(THONG_KE["kiem_dinh"]["muc_do"].get(y))
        hang_kd.append({"Yếu tố": YT[y]["ten"], "Tần suất: p-value": p1, "Tần suất: cải thiện CV": cv1,
                        "Tần suất: kết luận": kl1, "Mức BT: p-value": p2, "Mức BT: cải thiện CV": cv2,
                        "Mức BT: kết luận": kl2})
    st.dataframe(pd.DataFrame(hang_kd), hide_index=True)

    st.markdown("#### 4. Bảng hệ số (ước lượng trên toàn bộ dữ liệu, kèm khoảng tin cậy 95%)")

    def o_he_so(h, ktc, co):
        return f"{so(h, 3)} ({so(ktc[0], 2)}–{so(ktc[1], 2)})" if co and abs(h - 1) > 1e-12 else so(h, 3)

    hang_hs = []
    for y in BIEU_PHI["thu_tu_yeu_to"]:
        t = YT[y]
        for g in t["nhom"]:
            hang_hs.append({
                "Yếu tố": t["ten"], "Nhóm": t["nhan"][g] + (" (cơ sở)" if g == t["co_so"] else ""),
                "Hệ số tần suất": o_he_so(t["he_so_tan_suat"][g], t["ktc_tan_suat"][g], t["trong_mo_hinh"]["tan_suat"]),
                "Hệ số mức BT": o_he_so(t["he_so_muc_do"][g], t["ktc_muc_do"][g], t["trong_mo_hinh"]["muc_do"]),
                "Hệ số phí": so(t["he_so_phi"][g], 3)})
    st.dataframe(pd.DataFrame(hang_hs), hide_index=True, height=420)

    st.markdown("#### 5. Đánh giá trên tập kiểm tra (20% hợp đồng không dùng khi huấn luyện)")
    dg = THONG_KE["danh_gia"]
    ts = {r["mo_hinh"]: r for r in dg["tan_suat"]}
    md = {r["mo_hinh"]: r for r in dg["muc_do"]}
    pt = {r["mo_hinh"]: r for r in dg["phi_thuan"]}
    st.dataframe(pd.DataFrame([
        {"Mô hình": "Không dùng yếu tố nào", "D² tần suất": so(0, 4), "D² mức BT": so(0, 4),
         "Gini phí thuần": "—", "Phí dự báo / chi phí thực tế": "—"},
        {"Mô hình": "GLM (biểu phí trên web)", "D² tần suất": so(ts["GLM Poisson"]["d2"], 4),
         "D² mức BT": so(md["GLM Gamma"]["d2"], 4), "Gini phí thuần": so(pt["GLM (Poisson × Gamma)"]["gini"], 3),
         "Phí dự báo / chi phí thực tế": so(pt["GLM (Poisson × Gamma)"]["ty_le_du_bao_thuc_te"], 3)},
        {"Mô hình": "HistGradientBoosting", "D² tần suất": so(ts["HistGradientBoosting"]["d2"], 4),
         "D² mức BT": so(md["HistGradientBoosting"]["d2"], 4), "Gini phí thuần": so(pt["HistGradientBoosting"]["gini"], 3),
         "Phí dự báo / chi phí thực tế": so(pt["HistGradientBoosting"]["ty_le_du_bao_thuc_te"], 3)},
    ]), hide_index=True)
    st.caption("D²: tỷ lệ deviance giải thích được so với mô hình không dùng yếu tố nào (càng cao càng tốt). Gini: "
               "khả năng xếp hạng hợp đồng từ rủi ro thấp đến cao (0 = ngẫu nhiên). Chi phí tính trên số tiền đã chặn "
               f"ở {tien(BIEU_PHI['nguong_ton_that_lon'])}/vụ.")
    st.markdown(
        f"HistGradientBoosting chính xác hơn GLM một chút (Gini {so(pt['HistGradientBoosting']['gini'], 3)} so với "
        f"{so(pt['GLM (Poisson × Gamma)']['gini'], 3)}) nhưng không cho ra bảng hệ số. GLM được chọn làm biểu phí vì "
        "minh bạch, giải thích được và tính được bằng tay. Mức bồi thường gần như không dự báo được từ các yếu tố có "
        f"sẵn (D² ≈ 0): {phan_tram(THONG_KE['phan_phoi_boi_thuong']['ty_trong_vu_muc_khoan'], 0)} số vụ được chi trả "
        "quanh vài mức gần như cố định (1.128 €, 1.172 €, 1.204 €), nên khác biệt về phí chủ yếu đến từ tần suất.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Đường Lorenz: khả năng xếp hạng rủi ro**")
        st.image(anh_co_dinh("lorenz"))
    with c2:
        st.markdown("**Phí dự báo so với chi phí thực tế theo 10 nhóm rủi ro**")
        st.image(anh_co_dinh("thap_phan_vi"))

    st.markdown("#### 6. Hồ sơ minh hoạ (chi phí 20%, lợi nhuận 5%)")
    st.dataframe(pd.DataFrame([{
        "Hồ sơ": h["ten"], "Tần suất": phan_tram(h["tan_suat"], 1), "Mức BT kỳ vọng": tien(h["muc_bt"]),
        "Phí thuần / năm": tien(h["phi_thuan"]), "Phí bán / năm": tien(h["phi_ban"])}
        for h in THONG_KE["ho_so_mau"]]), hide_index=True)

st.caption(f"Dữ liệu: freMTPL2freq và freMTPL2sev (gói R CASdatasets, bản trên Kaggle). Mô hình tạo lúc "
           f"{THONG_KE['ngay_tao']} bằng scikit-learn {THONG_KE['phien_ban_sklearn']}. Kết quả chỉ phục vụ học tập.")
