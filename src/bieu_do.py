"""Biểu đồ cho web, vẽ bằng matplotlib (API hướng đối tượng, không dùng pyplot nên an toàn khi nhiều người
cùng mở web). Mỗi hàm trả về một Figure; dùng sang_png() để đổi sang ảnh PNG.

Quy ước màu (giống nhau trên mọi biểu đồ):
  xám  = số liệu thực tế          xanh = nhóm của khách hàng đang tính phí
  cam  = ảnh hưởng riêng theo GLM  đỏ / xanh = hệ số làm phí tăng / giảm
"""
import io
import math

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

from .dinh_dang import phan_tram, so

matplotlib.rcParams["font.family"] = "DejaVu Sans"   # có sẵn trong matplotlib, đủ dấu tiếng Việt

MAU = {
    "be_mat": "#fcfcfb", "chu": "#0b0b0b", "chu_phu": "#52514e", "chu_mo": "#898781",
    "luoi": "#e1e0d9", "truc": "#c3c2b7", "xam": "#c3c2b7",
    "nhan_manh": "#2a78d6", "du_bao": "#eb6834", "tang": "#e34948", "giam": "#2a78d6",
}
DPI = 200            # ảnh nét trên màn hình độ phân giải cao
PX_MOI_INCH = 100    # ảnh hiển thị khoảng 100 px mỗi inch -> dùng để quy đổi kích thước 24px, 4px, 2px
CO_CHU = 9
NET_DUONG = 1.44     # 2 px
CO_DIEM = 6          # đường kính ~8 px


# ---------------------------------------------------------------- tiện ích chung
def sang_png(fig: Figure) -> bytes:
    bo_dem = io.BytesIO()
    fig.savefig(bo_dem, format="png", dpi=DPI, facecolor=fig.get_facecolor())
    return bo_dem.getvalue()


def _hinh(rong, cao):
    return Figure(figsize=(rong, cao), dpi=DPI, facecolor=MAU["be_mat"])


def _truc(fig, vi_tri, ngang=False):
    ax = fig.add_axes(vi_tri, facecolor=MAU["be_mat"])
    for canh in ("top", "right", "left", "bottom"):
        ax.spines[canh].set_visible(False)
    goc = ax.spines["left" if ngang else "bottom"]
    goc.set_visible(True)
    goc.set_color(MAU["truc"])
    goc.set_linewidth(0.75)
    ax.tick_params(length=0, labelsize=CO_CHU, labelcolor=MAU["chu_phu"], pad=4)
    ax.set_axisbelow(True)
    (ax.xaxis if ngang else ax.yaxis).grid(True, color=MAU["luoi"], linewidth=0.6)
    return ax


def _px(ax):
    """Số đơn vị dữ liệu trên mỗi pixel theo trục x và y (cần đặt giới hạn trục trước)."""
    fig, khung = ax.figure, ax.get_position()
    rong_px = fig.get_figwidth() * khung.width * PX_MOI_INCH
    cao_px = fig.get_figheight() * khung.height * PX_MOI_INCH
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (x1 - x0) / rong_px, (y1 - y0) / cao_px


def _do_day(ax, ngang=False, toi_da_px=24, ti_le_khe=0.62):
    """Độ dày cột: không quá 24 px và không quá 62% khe."""
    px_x, px_y = _px(ax)
    return min(ti_le_khe, toi_da_px * (px_y if ngang else px_x))


def _cot(ax, vi_tri, gia_tri, do_day, mau, ngang=False, goc=0.0, ban_kinh_px=4):
    """Cột có đầu dữ liệu bo tròn 4 px, chân vuông nằm trên đường gốc."""
    px_x, px_y = _px(ax)
    nua = do_day / 2
    for c, v, m in zip(vi_tri, gia_tri, mau):
        if v is None or (isinstance(v, float) and math.isnan(v)) or v == goc:
            continue
        huong = 1.0 if v > goc else -1.0
        if ngang:
            r_doc = min(ban_kinh_px * px_x, abs(v - goc))
            r_ngang = min(ban_kinh_px * px_y, nua)
            dinh = [(goc, c - nua), (v - huong * r_doc, c - nua), (v, c - nua), (v, c - nua + r_ngang),
                    (v, c + nua - r_ngang), (v, c + nua), (v - huong * r_doc, c + nua), (goc, c + nua), (goc, c - nua)]
        else:
            r_doc = min(ban_kinh_px * px_y, abs(v - goc))
            r_ngang = min(ban_kinh_px * px_x, nua)
            dinh = [(c - nua, goc), (c - nua, v - huong * r_doc), (c - nua, v), (c - nua + r_ngang, v),
                    (c + nua - r_ngang, v), (c + nua, v), (c + nua, v - huong * r_doc), (c + nua, goc), (c - nua, goc)]
        ma = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
              Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(dinh, ma), facecolor=m, edgecolor="none", linewidth=0, zorder=2))


def _so_le_buoc(buoc: float) -> int:
    """Số chữ số thập phân đủ để ghi đúng bước chia trục."""
    for so_le in range(0, 4):
        if abs(round(buoc, so_le) - buoc) < 1e-9:
            return so_le
    return 3


def _truc_phan_tram(truc, toi_da, so_moc=4):
    dinh_vi = MaxNLocator(nbins=so_moc, steps=[1, 2, 2.5, 5, 10])
    moc = [m for m in dinh_vi.tick_values(0, toi_da) if 0 <= m <= toi_da * 1.0001]
    buoc = (moc[1] - moc[0]) * 100 if len(moc) > 1 else 1
    so_le = _so_le_buoc(buoc)
    truc.set_major_locator(FixedLocator(moc))
    truc.set_major_formatter(FuncFormatter(lambda v, _: f"{so(v * 100, so_le)}%"))


def _truc_euro(truc, toi_da, so_moc=4):
    dinh_vi = MaxNLocator(nbins=so_moc, steps=[1, 2, 2.5, 5, 10])
    moc = [m for m in dinh_vi.tick_values(0, toi_da) if 0 <= m <= toi_da * 1.0001]
    truc.set_major_locator(FixedLocator(moc))
    truc.set_major_formatter(FuncFormatter(lambda v, _: f"{so(v)} €"))


def _khung(fig, trai, duoi, rong, cao, ngang=False):
    """Tạo trục theo inch (trái, dưới, rộng, cao) để bố cục không phụ thuộc kích thước hình."""
    W, H = fig.get_figwidth(), fig.get_figheight()
    return _truc(fig, [trai / W, duoi / H, rong / W, cao / H], ngang=ngang)


def _chu_thich_khung(ax, noi_dung):
    """Tiêu đề nhỏ ngay trên góc trái của khung (cách 6 pt)."""
    ax.annotate(noi_dung, (0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                fontsize=CO_CHU, color=MAU["chu_phu"], ha="left", va="bottom")


def _chu_giai(fig, x_in, co_khach_hang, noi_duong):
    """Chú giải một hàng ở mép trên của hình (x_in: lề trái theo inch)."""
    tay_cam = [Patch(facecolor=MAU["xam"], label="Thực tế")]
    if co_khach_hang:
        tay_cam.append(Patch(facecolor=MAU["nhan_manh"], label="Nhóm của khách hàng"))
    tay_cam.append(Line2D([0], [0], color=MAU["du_bao"], lw=NET_DUONG if noi_duong else 0, marker="o",
                          ms=CO_DIEM, mec=MAU["be_mat"], mew=NET_DUONG, label="Ảnh hưởng riêng (GLM)"))
    W, H = fig.get_figwidth(), fig.get_figheight()
    fig.legend(handles=tay_cam, loc="upper left", bbox_to_anchor=(x_in / W, 1 - 0.06 / H), ncol=len(tay_cam),
               frameon=False, fontsize=CO_CHU, labelcolor=MAU["chu_phu"], handlelength=2.0, handleheight=0.9,
               columnspacing=1.6, borderaxespad=0, borderpad=0)


def _rong(danh_sach):
    return np.array([np.nan if v is None else float(v) for v in danh_sach])


def _chon_chi_tieu(hang, chi_tieu):
    if chi_tieu == "tan_suat":
        return {
            "thuc_te": _rong([h["tan_suat"] for h in hang]),
            "rieng": _rong([h["tan_suat_rieng"] for h in hang]),
            "ty_trong": _rong([h["ty_trong_exposure"] for h in hang]),
            "nhan_chinh": "Tần suất bồi thường (số vụ / năm bảo hiểm)",
            "nhan_phu": "Tỷ trọng số năm bảo hiểm",
            "ghi_gia_tri": lambda v: phan_tram(v, 1),
            "dat_truc": _truc_phan_tram,
        }
    return {
        "thuc_te": _rong([h["muc_bt"] for h in hang]),
        "rieng": _rong([h["muc_bt_rieng"] for h in hang]),
        "ty_trong": _rong([h["ty_trong_vu"] for h in hang]),
        "nhan_chinh": "Mức bồi thường trung bình (€ / vụ, đã chặn ở 100.000 €)",
        "nhan_phu": "Tỷ trọng số vụ",
        "ghi_gia_tri": lambda v: f"{so(v)} €",
        "dat_truc": _truc_euro,
    }


# ---------------------------------------------------------------- biểu đồ theo yếu tố
def ve_yeu_to(hang, nhan, chi_tieu="tan_suat", nhom_kh=None, co_thu_tu=True, ngang=False):
    """Hai khung chung trục: trên là chỉ tiêu (cột thực tế + điểm ảnh hưởng riêng), dưới là tỷ trọng."""
    if ngang:
        return _ve_ngang(hang, nhan, chi_tieu, nhom_kh)
    d = _chon_chi_tieu(hang, chi_tieu)
    n = len(hang)
    fig = _hinh(6.4, 4.0)
    ax = _khung(fig, 0.74, 1.62, 5.56, 1.74)      # khung chỉ tiêu
    ax2 = _khung(fig, 0.74, 0.40, 5.56, 0.62)     # khung tỷ trọng
    x = np.arange(n)
    for a in (ax, ax2):
        a.set_xlim(-0.6, n - 0.4)
    tran = np.nanmax(np.r_[d["thuc_te"], d["rieng"]]) * 1.2
    ax.set_ylim(0, tran)
    tran_phu = np.nanmax(d["ty_trong"]) * 1.15
    ax2.set_ylim(0, tran_phu)
    d["dat_truc"](ax.yaxis, tran)
    _truc_phan_tram(ax2.yaxis, tran_phu, so_moc=2)

    mau = [MAU["nhan_manh"] if h["nhom"] == nhom_kh else MAU["xam"] for h in hang]
    do_day = _do_day(ax)
    _cot(ax, x, d["thuc_te"], do_day, mau)
    _cot(ax2, x, d["ty_trong"], do_day, mau)
    ax.plot(x, d["rieng"], color=MAU["du_bao"], lw=NET_DUONG if co_thu_tu else 0, marker="o", ms=CO_DIEM,
            mec=MAU["be_mat"], mew=NET_DUONG, zorder=4, solid_capstyle="round", solid_joinstyle="round")

    # Chỉ ghi số cho nhóm của khách hàng
    for i, h in enumerate(hang):
        if h["nhom"] == nhom_kh and not math.isnan(d["thuc_te"][i]):
            dinh = np.nanmax([d["thuc_te"][i], d["rieng"][i]])
            ax.annotate(d["ghi_gia_tri"](d["thuc_te"][i]), (i, dinh), xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=CO_CHU, color=MAU["chu"], fontweight="bold", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax2.set_xticks(x)
    ax2.set_xticklabels([nhan[h["nhom"]] for h in hang])
    _chu_thich_khung(ax, d["nhan_chinh"])
    _chu_thich_khung(ax2, d["nhan_phu"])
    _chu_giai(fig, 0.70, nhom_kh is not None, co_thu_tu)
    return fig


def _ve_ngang(hang, nhan, chi_tieu, nhom_kh):
    """Bản nằm ngang cho yếu tố nhiều nhóm (vùng): sắp xếp theo chỉ tiêu thực tế, cao nhất ở trên."""
    d = _chon_chi_tieu(hang, chi_tieu)
    thu_tu = sorted(range(len(hang)), key=lambda i: -np.nan_to_num(d["thuc_te"][i], nan=-1.0))
    hang = [hang[i] for i in thu_tu]
    for k in ("thuc_te", "rieng", "ty_trong"):
        d[k] = d[k][thu_tu]
    n = len(hang)
    cao_khung = 0.27 * n
    fig = _hinh(12.8, cao_khung + 1.2)
    ax = _khung(fig, 2.2, 0.45, 6.8, cao_khung, ngang=True)
    ax2 = _khung(fig, 9.75, 0.45, 2.6, cao_khung, ngang=True)
    y = np.arange(n)[::-1].astype(float)
    for a in (ax, ax2):
        a.set_ylim(-0.6, n - 0.4)
    tran = np.nanmax(np.r_[d["thuc_te"], d["rieng"]]) * 1.15
    ax.set_xlim(0, tran)
    tran_phu = np.nanmax(d["ty_trong"]) * 1.12
    ax2.set_xlim(0, tran_phu)
    d["dat_truc"](ax.xaxis, tran, so_moc=5)
    _truc_phan_tram(ax2.xaxis, tran_phu, so_moc=3)

    mau = [MAU["nhan_manh"] if h["nhom"] == nhom_kh else MAU["xam"] for h in hang]
    do_day = _do_day(ax, ngang=True)
    _cot(ax, y, d["thuc_te"], do_day, mau, ngang=True)
    _cot(ax2, y, d["ty_trong"], do_day, mau, ngang=True)
    ax.plot(d["rieng"], y, lw=0, marker="o", ms=CO_DIEM, color=MAU["du_bao"], mec=MAU["be_mat"],
            mew=NET_DUONG, zorder=4)
    for i, h in enumerate(hang):
        if h["nhom"] == nhom_kh and not math.isnan(d["thuc_te"][i]):
            dau = np.nanmax([d["thuc_te"][i], d["rieng"][i]])
            ax.annotate(d["ghi_gia_tri"](d["thuc_te"][i]), (dau, y[i]), xytext=(7, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=CO_CHU, color=MAU["chu"], fontweight="bold", zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels([nhan[h["nhom"]] for h in hang])
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    _chu_thich_khung(ax, d["nhan_chinh"])
    _chu_thich_khung(ax2, d["nhan_phu"])
    _chu_giai(fig, 2.16, nhom_kh is not None, False)
    return fig


# ---------------------------------------------------------------- hệ số của khách hàng
def ve_he_so(chi_tiet):
    """Mỗi yếu tố làm phí tăng (đỏ) hay giảm (xanh) bao nhiêu lần so với hồ sơ cơ sở; thang log."""
    n = len(chi_tiet)
    fig = _hinh(7.2, 0.42 * n + 0.8)
    ax = _khung(fig, 2.55, 0.62, 4.45, 0.42 * n, ngang=True)
    ax.spines["left"].set_visible(False)
    gia_tri = [math.log(r["he_so_phi"]) for r in chi_tiet]
    bien = max(max(abs(v) for v in gia_tri), math.log(1.6)) * 1.45
    ax.set_xlim(-bien, bien)
    ax.set_ylim(-0.6, n - 0.4)
    y = np.arange(n)[::-1].astype(float)

    moc = [m for m in (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16) if abs(math.log(m)) <= bien]
    ax.xaxis.set_major_locator(FixedLocator([math.log(m) for m in moc]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"×{so(math.exp(v), 2).rstrip('0').rstrip(',')}"))
    ax.axvline(0, color=MAU["truc"], lw=0.75, zorder=1)

    mau = [MAU["tang"] if v > 1e-12 else MAU["giam"] for v in gia_tri]
    _cot(ax, y, gia_tri, _do_day(ax, ngang=True, toi_da_px=20), mau, ngang=True)
    for yi, v, r in zip(y, gia_tri, chi_tiet):
        if abs(v) <= 1e-12:
            ax.plot(0, yi, marker="o", ms=CO_DIEM, color=MAU["chu_mo"], mec=MAU["be_mat"], mew=NET_DUONG, zorder=3)
            ax.annotate("×1,00 (nhóm cơ sở)", (0, yi), xytext=(9, 0), textcoords="offset points", ha="left",
                        va="center", fontsize=CO_CHU, color=MAU["chu_phu"])
        else:
            ax.annotate(f"×{so(r['he_so_phi'], 2)}", (v, yi), xytext=(6 if v > 0 else -6, 0),
                        textcoords="offset points", ha="left" if v > 0 else "right", va="center",
                        fontsize=CO_CHU, color=MAU["chu"], fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['ten']}: {r['nhan']}" for r in chi_tiet], fontsize=CO_CHU, color=MAU["chu"])
    ax.set_xlabel("◀ làm phí giảm        hệ số so với hồ sơ cơ sở (thang log)        làm phí tăng ▶",
                  fontsize=CO_CHU - 0.5, color=MAU["chu_mo"], labelpad=6)
    return fig


# ---------------------------------------------------------------- phân phối số tiền bồi thường
def ve_phan_phoi(pp):
    moc = np.array(pp["moc_log10"], dtype=float)
    dem = np.array(pp["so_vu"], dtype=float)
    fig = _hinh(6.4, 4.0)
    ax = _khung(fig, 0.74, 0.42, 5.56, 3.1)
    ax.set_xlim(moc[0] - 0.05, moc[-1] + 0.05)
    ax.set_yscale("log")
    ax.set_ylim(0.7, dem.max() * 6)
    px_x, _ = _px(ax)
    tam, rong = (moc[:-1] + moc[1:]) / 2, (moc[1:] - moc[:-1]) - 2 * px_x
    co = dem > 0
    vuot = tam > math.log10(pp["nguong"])
    mau = np.where(vuot, MAU["tang"], MAU["nhan_manh"])
    ax.bar(tam[co], dem[co], width=rong[co], bottom=0.7, color=mau[co], linewidth=0, zorder=2)
    ax.xaxis.set_major_locator(FixedLocator(range(0, int(moc[-1]) + 1)))
    ten_moc = {0: "1 €", 1: "10 €", 2: "100 €", 3: "1.000 €", 4: "10.000 €", 5: "100.000 €", 6: "1 triệu €"}
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: ten_moc.get(int(round(v)), "")))
    ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000, 10000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: so(v)))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    nguong = math.log10(pp["nguong"])
    ax.axvline(nguong, color=MAU["chu_phu"], lw=0.9, zorder=3)
    ax.annotate(f"Ngưỡng tổn thất lớn\n{so(pp['nguong'])} €\n{pp['so_vu_vuot_nguong']} vụ vượt ngưỡng (đỏ)\n"
                f"= {phan_tram(pp['ty_trong_tien_vuot_nguong'], 1)} tổng tiền",
                (nguong, dem.max() * 4), xytext=(6, 0), textcoords="offset points", ha="left", va="top",
                fontsize=CO_CHU - 0.5, color=MAU["chu"], linespacing=1.35)
    dinh = tam[np.argmax(dem)]
    ax.annotate(f"{phan_tram(pp['ty_trong_vu_muc_khoan'], 0)} số vụ ở mức\n1.100–1.250 €",
                (dinh, dem.max()), xytext=(-10, 2), textcoords="offset points", ha="right", va="bottom",
                fontsize=CO_CHU - 0.5, color=MAU["chu"])
    _chu_thich_khung(ax, f"Số vụ (thang log) · trung vị {so(pp['trung_vi'])} €, trung bình {so(pp['trung_binh'])} €")
    return fig


# ---------------------------------------------------------------- đánh giá mô hình
def ve_lorenz(lorenz, danh_gia_phi):
    x = np.array(lorenz["x"])
    fig = _hinh(6.4, 4.4)
    ax = _khung(fig, 0.74, 0.72, 5.36, 2.98)
    ax.xaxis.grid(True, color=MAU["luoi"], linewidth=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot(x, x, color=MAU["chu_mo"], lw=0.9, label="Xếp ngẫu nhiên")
    gini = {r["mo_hinh"]: r["gini"] for r in danh_gia_phi}
    ax.plot(x, lorenz["GLM"], color=MAU["nhan_manh"], lw=NET_DUONG, solid_capstyle="round",
            label=f"GLM (Gini {so(gini['GLM (Poisson × Gamma)'], 3)})")
    ax.plot(x, lorenz["HistGradientBoosting"], color=MAU["du_bao"], lw=NET_DUONG, solid_capstyle="round",
            label=f"HistGradientBoosting (Gini {so(gini['HistGradientBoosting'], 3)})")
    for truc in (ax.xaxis, ax.yaxis):
        truc.set_major_locator(FixedLocator([0, 0.25, 0.5, 0.75, 1]))
        truc.set_major_formatter(FuncFormatter(lambda v, _: f"{so(v * 100)}%"))
    ax.set_xlabel("Tỷ lệ tích luỹ số năm bảo hiểm (từ rủi ro dự báo thấp đến cao)", fontsize=CO_CHU,
                  color=MAU["chu_phu"], labelpad=6)
    _chu_thich_khung(ax, "Tỷ lệ tích luỹ chi phí bồi thường thực tế")
    fig.legend(loc="upper left", bbox_to_anchor=(0.70 / 6.4, 1 - 0.06 / 4.4), ncol=3, frameon=False,
               fontsize=CO_CHU - 0.5, labelcolor=MAU["chu_phu"], handlelength=1.6, columnspacing=1.2,
               borderaxespad=0, borderpad=0)
    return fig


def ve_thap_phan_vi(bang):
    n = len(bang)
    x = np.arange(1, n + 1)
    thuc_te = np.array([r["thuc_te"] for r in bang])
    du_bao = np.array([r["du_bao"] for r in bang])
    fig = _hinh(6.4, 4.4)
    ax = _khung(fig, 0.82, 0.72, 5.42, 2.98)
    ax.set_xlim(0.4, n + 0.6)
    tran = max(thuc_te.max(), du_bao.max()) * 1.12
    ax.set_ylim(0, tran)
    _truc_euro(ax.yaxis, tran, so_moc=5)
    _cot(ax, x, thuc_te, _do_day(ax), [MAU["xam"]] * n)
    ax.plot(x, du_bao, color=MAU["du_bao"], lw=NET_DUONG, marker="o", ms=CO_DIEM, mec=MAU["be_mat"],
            mew=NET_DUONG, zorder=4, solid_capstyle="round", solid_joinstyle="round")
    ax.set_xticks(x)
    ax.set_xlabel("Nhóm rủi ro theo phí dự báo (1 = thấp nhất; mỗi nhóm 10% số năm bảo hiểm)", fontsize=CO_CHU,
                  color=MAU["chu_phu"], labelpad=6)
    _chu_thich_khung(ax, "Phí thuần (€ / năm, chi phí đã chặn ở 100.000 €/vụ)")
    tay_cam = [Patch(facecolor=MAU["xam"], label="Chi phí thực tế"),
               Line2D([0], [0], color=MAU["du_bao"], lw=NET_DUONG, marker="o", ms=CO_DIEM, mec=MAU["be_mat"],
                      mew=NET_DUONG, label="Phí dự báo (GLM)")]
    fig.legend(handles=tay_cam, loc="upper left", bbox_to_anchor=(0.78 / 6.4, 1 - 0.06 / 4.4), ncol=2,
               frameon=False, fontsize=CO_CHU, labelcolor=MAU["chu_phu"], handlelength=2.0, columnspacing=1.6,
               borderaxespad=0, borderpad=0)
    return fig
