"""Cấu hình chung cho toàn bộ dự án: đường dẫn, tham số làm sạch và cách chia nhóm.

Mọi con số "chốt tay" của bài đều nằm ở đây để dễ giải thích và dễ thay đổi.
"""
from pathlib import Path

# ---------------------------------------------------------------- Đường dẫn
THU_MUC_GOC = Path(__file__).resolve().parent.parent
THU_MUC_DU_LIEU = THU_MUC_GOC / "data"
THU_MUC_MO_HINH = THU_MUC_GOC / "models"

FILE_TAN_SUAT = THU_MUC_DU_LIEU / "freMTPL2freq.csv"     # đặc điểm hợp đồng + số vụ
FILE_BOI_THUONG = THU_MUC_DU_LIEU / "freMTPL2sev.csv"    # số tiền từng vụ bồi thường

FILE_BIEU_PHI = THU_MUC_MO_HINH / "bieu_phi.json"        # phí cơ sở + hệ số (web dùng để tính phí)
FILE_THONG_KE = THU_MUC_MO_HINH / "thong_ke.json"        # số liệu tổng hợp cho phần phân tích

# ---------------------------------------------------------------- Làm sạch
NGUONG_TON_THAT_LON = 100_000   # € / vụ. Phần vượt ngưỡng được phân bổ đều vào phí.
SO_VU_TOI_DA = 4                # số vụ / hợp đồng tối đa đưa vào mô hình tần suất
EXPOSURE_TOI_DA = 1.0           # thời gian bảo hiểm tối đa (năm)

# ---------------------------------------------------------------- Mô hình
TY_LE_KIEM_TRA = 0.2            # 20% hợp đồng để đánh giá mô hình
HAT_GIONG = 42                  # cố định để chạy lại ra cùng kết quả
MUC_Y_NGHIA = 0.05              # ngưỡng giữ/loại yếu tố khi chọn biến
SO_PHAN_CV = 5                  # kiểm định chéo 5 phần ...
SO_LAN_LAP_CV = 4               # ... lặp lại 4 lần với cách chia khác nhau

# ---------------------------------------------------------------- Yếu tố tính phí
# kieu = "so": chia theo khoảng, mỗi nhóm là (nhãn, cận trên). Cận trên None = không giới hạn.
#              Một giá trị thuộc nhóm đầu tiên có cận trên >= giá trị đó.
# kieu = "danh_muc": mỗi giá trị là một nhóm; nhom = None nghĩa là lấy từ dữ liệu.
YEU_TO = {
    "DrivAge": {
        "ten": "Tuổi người lái",
        "kieu": "so",
        "don_vi": "tuổi",
        "nhom": [("18–20", 20), ("21–25", 25), ("26–30", 30), ("31–40", 40),
                 ("41–50", 50), ("51–60", 60), ("61–70", 70), ("≥ 71", None)],
    },
    "BonusMalus": {
        "ten": "Hệ số BonusMalus",
        "kieu": "so",
        "don_vi": "",
        "nhom": [("50", 50), ("51–60", 60), ("61–70", 70), ("71–80", 80), ("81–90", 90),
                 ("91–99", 99), ("100", 100), ("101–120", 120), ("> 120", None)],
    },
    "VehAge": {
        "ten": "Tuổi xe",
        "kieu": "so",
        "don_vi": "năm",
        "nhom": [("0–2", 2), ("3–5", 5), ("6–10", 10), ("11–15", 15), ("16–20", 20), ("> 20", None)],
    },
    "VehPower": {
        "ten": "Công suất xe",
        "kieu": "so",
        "don_vi": "",
        "nhom": [("4", 4), ("5", 5), ("6", 6), ("7", 7), ("8", 8), ("9", 9), ("10", 10), ("≥ 11", None)],
    },
    "VehBrand": {
        "ten": "Hãng xe",
        "kieu": "danh_muc",
        "nhom": ["B1", "B2", "B3", "B4", "B5", "B6", "B10", "B11", "B12", "B13", "B14"],
    },
    "VehGas": {
        "ten": "Nhiên liệu",
        "kieu": "danh_muc",
        "nhom": ["Regular", "Diesel"],
        "nhan": {"Regular": "Xăng", "Diesel": "Dầu diesel"},
    },
    "Density": {
        "ten": "Mật độ dân số",
        "kieu": "so",
        "don_vi": "người/km²",
        "nhom": [("≤ 50", 50), ("51–100", 100), ("101–500", 500), ("501–2.000", 2000),
                 ("2.001–10.000", 10000), ("> 10.000", None)],
    },
    "Region": {
        "ten": "Vùng",
        "kieu": "danh_muc",
        "nhom": None,
    },
}

# Thứ tự hiển thị trên web và trong bảng hệ số
THU_TU_YEU_TO = ["DrivAge", "BonusMalus", "VehAge", "VehPower", "VehBrand", "VehGas", "Density", "Region"]


def cot_nhom(yeu_to: str) -> str:
    """Tên cột chứa nhóm của một yếu tố, ví dụ DrivAge -> nhom_DrivAge."""
    return f"nhom_{yeu_to}"
