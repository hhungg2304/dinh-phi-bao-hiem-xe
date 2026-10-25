"""Định dạng số kiểu Việt Nam: dấu chấm ngăn hàng nghìn, dấu phẩy thập phân."""


def so(x: float, so_le: int = 0) -> str:
    s = f"{x:,.{so_le}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def phan_tram(x: float, so_le: int = 1) -> str:
    """0,0738 -> '7,4%'"""
    return f"{so(x * 100, so_le)}%"


def tien(x: float, so_le: int = 0) -> str:
    return f"{so(x, so_le)} €"


def he_so(x: float, so_le: int = 2) -> str:
    return f"×{so(x, so_le)}"


def p_value(p: float) -> str:
    if p < 0.001:
        return "< 0,001"
    return so(p, 3)
