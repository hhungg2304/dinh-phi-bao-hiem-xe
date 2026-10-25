"""Đọc, nối và làm sạch dữ liệu freMTPL2, rồi chia nhóm các yếu tố tính phí."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import cau_hinh as ch

COT_TAN_SUAT = ["IDpol", "ClaimNb", "Exposure", "VehPower", "VehAge", "DrivAge",
                "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region"]
COT_BOI_THUONG = ["IDpol", "ClaimAmount"]


def _so(x, so_le=0):
    """Định dạng số kiểu Việt Nam cho nhật ký: 1.234,5"""
    s = f"{x:,.{so_le}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def doc_du_lieu(file_tan_suat=ch.FILE_TAN_SUAT, file_boi_thuong=ch.FILE_BOI_THUONG):
    """Đọc hai file CSV và kiểm tra đủ cột."""
    for duong_dan in (file_tan_suat, file_boi_thuong):
        if not Path(duong_dan).exists():
            raise FileNotFoundError(
                f"Không tìm thấy {duong_dan}. Tải bộ dữ liệu freMTPL2 trên Kaggle và đặt "
                "freMTPL2freq.csv, freMTPL2sev.csv vào thư mục data/ (xem README)."
            )
    tan_suat = pd.read_csv(file_tan_suat)
    boi_thuong = pd.read_csv(file_boi_thuong)

    for ten, df, cot_can in (("freMTPL2freq", tan_suat, COT_TAN_SUAT), ("freMTPL2sev", boi_thuong, COT_BOI_THUONG)):
        thieu = [c for c in cot_can if c not in df.columns]
        if thieu:
            raise ValueError(f"File {ten} thiếu cột: {thieu}")

    # Một số bản sao trên mạng (ví dụ OpenML) bọc chuỗi trong dấu nháy đơn: 'B12'
    for cot in ["VehBrand", "VehGas", "Area", "Region"]:
        tan_suat[cot] = tan_suat[cot].astype(str).str.strip("'")
    tan_suat["IDpol"] = tan_suat["IDpol"].astype("int64")
    boi_thuong["IDpol"] = boi_thuong["IDpol"].astype("int64")
    return tan_suat, boi_thuong


def xep_nhom(gia_tri: pd.Series, yeu_to: str, cac_nhom: list) -> pd.Categorical:
    """Gán mỗi giá trị vào nhóm của yếu tố (cùng quy tắc với src/bieu_phi.py)."""
    cau_hinh = ch.YEU_TO[yeu_to]
    if cau_hinh["kieu"] == "so":
        nhan = [n for n, _ in cau_hinh["nhom"]]
        moc = [-np.inf] + [np.inf if t is None else t for _, t in cau_hinh["nhom"]]
        return pd.cut(gia_tri, bins=moc, labels=nhan, right=True)
    return pd.Categorical(gia_tri, categories=cac_nhom)


def danh_sach_nhom(yeu_to: str, du_lieu: pd.DataFrame) -> list:
    """Danh sách nhóm theo thứ tự hiển thị."""
    cau_hinh = ch.YEU_TO[yeu_to]
    if cau_hinh["kieu"] == "so":
        return [n for n, _ in cau_hinh["nhom"]]
    if cau_hinh["nhom"] is not None:
        return list(cau_hinh["nhom"])
    return sorted(du_lieu[yeu_to].unique().tolist())


def lam_sach(tan_suat: pd.DataFrame, boi_thuong: pd.DataFrame):
    """Trả về (bảng hợp đồng, bảng từng vụ, danh sách nhóm của từng yếu tố, nhật ký các bước)."""
    nhat_ky = []
    tan_suat = tan_suat.sort_values("IDpol").reset_index(drop=True)

    so_hd_goc = len(tan_suat)
    nhat_ky.append({
        "buoc": "Dữ liệu gốc",
        "chi_tiet": f"{_so(so_hd_goc)} hợp đồng, {_so(tan_suat['ClaimNb'].sum())} vụ ghi trong freMTPL2freq; "
                    f"{_so(len(boi_thuong))} vụ có số tiền trong freMTPL2sev "
                    f"(tổng {_so(boi_thuong['ClaimAmount'].sum() / 1e6, 1)} triệu €).",
    })

    # 1. Khối đầu file: các hợp đồng xếp trước hợp đồng không-bồi-thường đầu tiên đều có bồi thường,
    #    và gần như không có số tiền trong file sev -> mẫu bị chọn lệch, loại bỏ.
    vi_tri_khong_bt = np.flatnonzero(tan_suat["ClaimNb"].to_numpy() == 0)
    id_bat_dau = int(tan_suat["IDpol"].iloc[vi_tri_khong_bt[0]]) if len(vi_tri_khong_bt) else int(tan_suat["IDpol"].min())
    khoi_dau = tan_suat["IDpol"] < id_bat_dau
    if khoi_dau.any():
        id_khoi = tan_suat.loc[khoi_dau, "IDpol"]
        ty_le_co_tien = id_khoi.isin(boi_thuong["IDpol"]).mean()
        nhat_ky.append({
            "buoc": "Loại khối dữ liệu lệch",
            "chi_tiet": f"{_so(khoi_dau.sum())} hợp đồng đầu file (IDpol < {_so(id_bat_dau)}) đều có bồi thường "
                        f"nhưng chỉ {_so(ty_le_co_tien * 100, 1)}% có số tiền trong freMTPL2sev. Đây là mẫu chỉ gồm "
                        "hợp đồng có tai nạn, không đại diện cho danh mục, nên bị loại.",
        })
    tan_suat = tan_suat.loc[~khoi_dau].copy()

    # 2. Ghép số tiền vào hợp đồng
    thuoc_danh_muc = boi_thuong["IDpol"].isin(tan_suat["IDpol"])
    mo_coi = ~boi_thuong["IDpol"].isin(tan_suat["IDpol"]) & (boi_thuong["IDpol"] >= id_bat_dau)
    if mo_coi.any():
        nhat_ky.append({
            "buoc": "Bỏ bồi thường không có hợp đồng",
            "chi_tiet": f"{_so(mo_coi.sum())} vụ ({_so(boi_thuong.loc[mo_coi, 'IDpol'].nunique())} mã hợp đồng, "
                        f"{_so(boi_thuong.loc[mo_coi, 'ClaimAmount'].sum())} €) không tìm thấy trong freMTPL2freq.",
        })
    vu = boi_thuong.loc[thuoc_danh_muc].copy()
    vu["TienChan"] = vu["ClaimAmount"].clip(upper=ch.NGUONG_TON_THAT_LON)
    vuot = vu["ClaimAmount"] > ch.NGUONG_TON_THAT_LON
    nhat_ky.append({
        "buoc": "Chặn tổn thất lớn",
        "chi_tiet": f"{_so(vuot.sum())} vụ vượt {_so(ch.NGUONG_TON_THAT_LON)} € (lớn nhất "
                    f"{_so(vu['ClaimAmount'].max())} €). Mô hình mức độ dùng số tiền đã chặn; phần vượt "
                    f"({_so((vu['ClaimAmount'] - vu['TienChan']).sum() / vu['ClaimAmount'].sum() * 100, 1)}% tổng tiền) "
                    "được phân bổ đều vào phí qua hệ số điều chỉnh.",
    })
    theo_hd = vu.groupby("IDpol").agg(SoVu=("ClaimAmount", "size"),
                                      TienBT=("ClaimAmount", "sum"),
                                      TienBTChan=("TienChan", "sum"))
    df = tan_suat.join(theo_hd, on="IDpol")
    df[["SoVu", "TienBT", "TienBTChan"]] = df[["SoVu", "TienBT", "TienBTChan"]].fillna(0.0)

    # 3. Số vụ dùng cho mô hình = số vụ có số tiền (để tần suất và mức độ khớp nhau)
    co_vu_khong_tien = (df["ClaimNb"] > 0) & (df["SoVu"] == 0)
    if co_vu_khong_tien.any():
        nhat_ky.append({
            "buoc": "Vụ không có số tiền",
            "chi_tiet": f"{_so(co_vu_khong_tien.sum())} hợp đồng còn lại có ClaimNb > 0 nhưng không có số tiền: "
                        "coi như không phát sinh chi phí.",
        })
    qua_nhieu = df["SoVu"] > ch.SO_VU_TOI_DA
    df["SoVuMH"] = df["SoVu"].clip(upper=ch.SO_VU_TOI_DA)
    nhat_ky.append({
        "buoc": "Giới hạn số vụ",
        "chi_tiet": f"{_so(qua_nhieu.sum())} hợp đồng có trên {ch.SO_VU_TOI_DA} vụ (nhiều nhất "
                    f"{_so(df['SoVu'].max())} vụ, nghi là xe đội/lỗi nhập) được tính {ch.SO_VU_TOI_DA} vụ "
                    "trong mô hình tần suất.",
    })

    # 4. Thời gian bảo hiểm tối đa 1 năm
    qua_1_nam = df["Exposure"] > ch.EXPOSURE_TOI_DA
    df["Exposure"] = df["Exposure"].clip(upper=ch.EXPOSURE_TOI_DA)
    nhat_ky.append({
        "buoc": "Giới hạn thời gian bảo hiểm",
        "chi_tiet": f"{_so(qua_1_nam.sum())} hợp đồng có Exposure > 1 năm (tối đa 2,01) được đưa về 1 năm.",
    })

    # 5. Chia nhóm
    cac_nhom = {}
    for yeu_to in ch.THU_TU_YEU_TO:
        cac_nhom[yeu_to] = danh_sach_nhom(yeu_to, df)
        df[ch.cot_nhom(yeu_to)] = xep_nhom(df[yeu_to], yeu_to, cac_nhom[yeu_to])
        if df[ch.cot_nhom(yeu_to)].isna().any():
            raise ValueError(f"Có giá trị {yeu_to} không thuộc nhóm nào – kiểm tra cấu hình nhóm.")

    nhat_ky.append({
        "buoc": "Dữ liệu dùng cho mô hình",
        "chi_tiet": f"{_so(len(df))} hợp đồng, {_so(df['Exposure'].sum())} năm bảo hiểm, "
                    f"{_so(df['SoVu'].sum())} vụ, tổng {_so(df['TienBT'].sum() / 1e6, 1)} triệu €.",
    })

    cot_giu = (["IDpol", "Exposure", "ClaimNb", "SoVu", "SoVuMH", "TienBT", "TienBTChan"]
               + ch.THU_TU_YEU_TO + [ch.cot_nhom(y) for y in ch.THU_TU_YEU_TO])
    return df[cot_giu].reset_index(drop=True), vu.reset_index(drop=True), cac_nhom, nhat_ky
