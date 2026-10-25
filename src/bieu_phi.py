"""Tính phí từ biểu phí đã xuất (models/bieu_phi.json).

Chỉ dùng Python thuần nên web không cần scikit-learn:

    Phí thuần = Phí cơ sở × hệ số(tuổi lái) × hệ số(BonusMalus) × ... × hệ số(vùng)
    Phí bán   = Phí thuần / (1 − tỷ lệ chi phí − tỷ lệ lợi nhuận)
"""
import json
from pathlib import Path


def doc_bieu_phi(duong_dan) -> dict:
    return json.loads(Path(duong_dan).read_text(encoding="utf-8"))


def xep_nhom(bieu_phi: dict, yeu_to: str, gia_tri) -> str:
    """Trả về mã nhóm của một giá trị (cùng quy tắc với lúc huấn luyện)."""
    tt = bieu_phi["yeu_to"][yeu_to]
    if tt["kieu"] == "so":
        for nhom, can_tren in zip(tt["nhom"], tt["can_tren"]):
            if can_tren is None or gia_tri <= can_tren:
                return nhom
        raise ValueError(f"{tt['ten']}: giá trị {gia_tri} không thuộc nhóm nào")
    gia_tri = str(gia_tri)
    if gia_tri not in tt["nhom"]:
        raise ValueError(f"{tt['ten']}: không có nhóm '{gia_tri}'")
    return gia_tri


def tinh_phi(bieu_phi: dict, ho_so: dict, ty_le_chi_phi: float = 0.0, ty_le_loi_nhuan: float = 0.0) -> dict:
    """Tính phí cho một hồ sơ, ví dụ ho_so = {"DrivAge": 35, "BonusMalus": 50, ..., "Region": "Centre"}."""
    phu_phi = ty_le_chi_phi + ty_le_loi_nhuan
    if not 0 <= phu_phi < 1:
        raise ValueError("Tổng tỷ lệ chi phí và lợi nhuận phải nằm trong [0, 1)")

    co_so = bieu_phi["phi_co_so"]
    tan_suat, muc_bt = co_so["tan_suat"], co_so["muc_bt"]
    chi_tiet = []
    for yt in bieu_phi["thu_tu_yeu_to"]:
        tt = bieu_phi["yeu_to"][yt]
        nhom = xep_nhom(bieu_phi, yt, ho_so[yt])
        h_ts, h_md = tt["he_so_tan_suat"][nhom], tt["he_so_muc_do"][nhom]
        tan_suat *= h_ts
        muc_bt *= h_md
        chi_tiet.append({"yeu_to": yt, "ten": tt["ten"], "nhom": nhom, "nhan": tt["nhan"][nhom],
                         "he_so_tan_suat": h_ts, "he_so_muc_do": h_md, "he_so_phi": h_ts * h_md})

    phi_thuan = tan_suat * muc_bt
    return {
        "tan_suat": tan_suat,                      # số vụ kỳ vọng / năm
        "muc_bt": muc_bt,                          # € kỳ vọng / vụ (đã gồm phân bổ tổn thất lớn)
        "phi_thuan": phi_thuan,                    # € / năm
        "phi_ban": phi_thuan / (1 - phu_phi),      # € / năm
        "phi_co_so": co_so["phi_thuan"],
        "tich_he_so": phi_thuan / co_so["phi_thuan"],
        "chi_tiet": chi_tiet,
    }
