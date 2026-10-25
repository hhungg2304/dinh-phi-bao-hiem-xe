# Định phí bảo hiểm TNDS xe cơ giới bằng GLM

Bài này xây dựng biểu phí bảo hiểm trách nhiệm dân sự xe cơ giới từ dữ liệu thực tế **freMTPL2**, gồm 678.013 hợp
đồng của một công ty bảo hiểm Pháp. Mô hình gồm hai phần:

- **Tần suất** (mỗi năm có bao nhiêu vụ): GLM Poisson
- **Mức bồi thường** (mỗi vụ tốn bao nhiêu tiền): GLM Gamma

Kết quả là một **công thức tính phí** dạng nhân:

```
Phí thuần = Phí cơ sở × hệ số(tuổi lái) × hệ số(BonusMalus) × hệ số(tuổi xe) × … × hệ số(vùng)
Phí bán   = Phí thuần / (1 − tỷ lệ chi phí − tỷ lệ lợi nhuận)
```

Web Streamlit chỉ có một trang, gồm ba phần: tính phí cho một khách hàng, biểu đồ phân tích dữ liệu (nhóm của khách
hàng được tô nổi bật) và chi tiết mô hình.

## Kết quả chính

| Chỉ tiêu | Giá trị |
|---|---|
| Hợp đồng dùng cho mô hình | 668.626 (353.560 năm bảo hiểm, 26.137 vụ) |
| Tần suất bồi thường trung bình | 7,38% / năm |
| Mức bồi thường trung bình | 2.267 € / vụ |
| Phí thuần trung bình | 168 € / năm |
| Phí cơ sở (hồ sơ cơ sở) | 108,03 € / năm |

- **Hồ sơ cơ sở** là hồ sơ có hệ số = 1, gồm các nhóm đông khách nhất: tuổi lái 41–50, BonusMalus 50, tuổi xe 6–10,
  công suất 6, hãng B1, xe chạy xăng, mật độ dân số 101–500 người/km², vùng Centre.
- **BonusMalus là yếu tố mạnh nhất.** So với hệ số 50, nhóm 100 (mới tham gia) có tần suất gấp 4,8 lần, nhóm trên 120
  gấp 10,1 lần.
- **Các yếu tố khác về tần suất:** xe chạy dầu diesel ×1,17; mật độ dân số trên 10.000 người/km² ×1,28; xe trên 20 năm
  tuổi ×0,55; hãng B12 ×0,76.
- **Tuổi người lái:** sau khi đã tính BonusMalus, người lái trẻ không còn rủi ro cao như số liệu thô cho thấy. Nhóm
  18–20 tuổi có tần suất thực tế 23,9%, nhưng ảnh hưởng riêng của tuổi chỉ khoảng 9,9%, vì người mới lái đều có
  BonusMalus cao.
- **Mức bồi thường gần như không dự báo được.** Chỉ tuổi người lái được giữ lại: nhóm 18–20 ×1,72, nhóm từ 71 tuổi
  ×1,38. Lý do là 43% số vụ được chi trả quanh vài mức gần như cố định, từ 1.128 € đến 1.204 €.
- **Đánh giá trên tập kiểm tra (20% hợp đồng):** GLM có Gini 0,292, HistGradientBoosting có Gini 0,305. Mô hình học
  máy chính xác hơn một chút nhưng không cho ra được bảng hệ số minh bạch như GLM.

| Hồ sơ minh hoạ | Phí thuần / năm | So với trung bình |
|---|---|---|
| 20 tuổi, BonusMalus 100, sống ở thành phố (Île-de-France) | 942 € | ×5,6 |
| 45 tuổi, BonusMalus 50, sống ở nông thôn (Bretagne) | 107 € | ×0,64 |
| 72 tuổi, BonusMalus 50, sống ở đô thị đông dân (PACA) | 171 € | ×1,02 |

## Cấu trúc thư mục

```
dinh-phi-bao-hiem-xe/
├── app.py                  # Web Streamlit (một trang)
├── src/
│   ├── cau_hinh.py         # Tham số: đường dẫn, ngưỡng tổn thất lớn, cách chia nhóm
│   ├── xu_ly_du_lieu.py    # Đọc, nối, làm sạch dữ liệu, chia nhóm
│   ├── mo_hinh.py          # GLM: ma trận thiết kế, sai số chuẩn, kiểm định, chọn yếu tố
│   ├── huan_luyen.py       # Chạy toàn bộ quy trình và xuất kết quả ra models/
│   ├── bieu_phi.py         # Hàm tinh_phi() – tính phí từ biểu phí (Python thuần)
│   ├── bieu_do.py          # Biểu đồ matplotlib cho web
│   └── dinh_dang.py        # Định dạng số kiểu Việt Nam (1.234,5)
├── models/
│   ├── bieu_phi.json       # Phí cơ sở + hệ số từng nhóm (web dùng để tính phí)
│   └── thong_ke.json       # Số liệu tổng hợp, kiểm định, đánh giá (web dùng để vẽ biểu đồ)
├── data/                   # Đặt freMTPL2freq.csv, freMTPL2sev.csv vào đây (không đưa lên GitHub)
├── .streamlit/config.toml  # Giao diện sáng, màu chủ đạo
└── requirements.txt
```

## Chạy trên máy

```bash
pip install -r requirements.txt

# Mở web (dùng kết quả có sẵn trong models/)
streamlit run app.py

# (Tuỳ chọn) Huấn luyện lại – cần 2 file dữ liệu trong data/, mất khoảng 2 phút
python -m src.huan_luyen
```

## Đưa lên web miễn phí (Streamlit Community Cloud)

1. Tạo một repo mới trên GitHub, rồi đẩy toàn bộ thư mục lên. Hai file CSV trong `data/` tự động bị bỏ qua.

   ```bash
   git init
   git add .
   git commit -m "Định phí bảo hiểm xe cơ giới bằng GLM"
   git branch -M main
   git remote add origin https://github.com/<tai-khoan>/dinh-phi-bao-hiem-xe.git
   git push -u origin main
   ```

2. Vào https://share.streamlit.io và đăng nhập bằng tài khoản GitHub. Chọn **Create app**, chọn repo, nhánh `main`,
   file `app.py`, rồi bấm **Deploy**.
3. Sau vài phút, bạn nhận được link dạng `https://<ten-app>.streamlit.app`.

Web chỉ đọc `models/*.json`, nên không cần dữ liệu gốc trên GitHub. Khi huấn luyện lại, chỉ cần commit hai file JSON
mới là web tự cập nhật.

## Dữ liệu

Hai bảng `freMTPL2freq` và `freMTPL2sev` lấy từ gói R CASdatasets (Dutang & Charpentier). Trên Kaggle, tìm theo tên
"freMTPL2". Có thể tải cùng dữ liệu từ OpenML: [freMTPL2freq](https://www.openml.org/d/41214),
[freMTPL2sev](https://www.openml.org/d/41215).

| Cột | Ý nghĩa |
|---|---|
| IDpol | Mã hợp đồng (dùng để nối hai bảng) |
| ClaimNb | Số vụ bồi thường |
| Exposure | Thời gian được bảo hiểm (năm) |
| VehPower, VehAge | Công suất xe, tuổi xe |
| DrivAge | Tuổi người lái |
| BonusMalus | Hệ số thưởng/phạt: 100 khi mới tham gia, thấp nhất 50, trên 100 là bị phạt |
| VehBrand, VehGas | Hãng xe (mã ẩn danh), nhiên liệu |
| Area, Density | Mã khu vực, mật độ dân số nơi người lái sống |
| Region | Vùng của Pháp |
| ClaimAmount | Số tiền bồi thường từng vụ (bảng sev) |

## Phương pháp

**1. Làm sạch dữ liệu** (chi tiết hiện trên web, mục *Chi tiết mô hình*)

- Loại 9.387 hợp đồng đầu file (IDpol < 24.952). Khối này chỉ gồm hợp đồng có bồi thường, và 97% trong số đó không có
  số tiền trong bảng sev, nên là mẫu bị chọn lệch.
- Chặn mỗi vụ bồi thường ở 100.000 €. Có 41 vụ vượt ngưỡng, chiếm 17,9% tổng tiền; phần vượt được phân bổ đều vào phí.
- Exposure tối đa 1 năm. Số vụ mỗi hợp đồng tối đa 4 khi đưa vào mô hình tần suất.
- Số vụ dùng cho mô hình là số vụ có số tiền, để tần suất và mức bồi thường khớp nhau.

**2. Chia nhóm.** Tuổi lái, BonusMalus, tuổi xe, công suất và mật độ được chia khoảng theo mức thay đổi của tần suất
trong dữ liệu (xem `src/cau_hinh.py`). Hãng xe, nhiên liệu và vùng giữ nguyên từng giá trị. Nhóm cơ sở của mỗi yếu tố là
nhóm có nhiều năm bảo hiểm nhất.

**3. Mô hình**

- Tần suất: GLM Poisson, hàm liên kết log. Biến mục tiêu là số vụ / exposure, trọng số là exposure (tương đương offset
  log(exposure)).
- Mức bồi thường: GLM Gamma, hàm liên kết log. Biến mục tiêu là số tiền trung bình mỗi vụ (đã chặn), trọng số là số vụ.
- Hệ số của một nhóm = exp(β). Sai số chuẩn tính từ ma trận thông tin Fisher. Riêng mô hình Gamma, tham số phân tán
  được ước lượng bằng Pearson.

**4. Chọn yếu tố** (loại lùi dần trên 80% dữ liệu huấn luyện). Một yếu tố chỉ được giữ khi thoả cả hai điều kiện:

- Có ý nghĩa thống kê: kiểm định likelihood ratio cho p ≤ 5%.
- Giúp dự báo tốt hơn rõ rệt: bỏ yếu tố đi làm deviance kiểm định chéo (5 phần × 4 lần) tăng hơn 1 sai số chuẩn.

Kết quả: tần suất giữ cả 8 yếu tố. Mức bồi thường chỉ giữ tuổi người lái; BonusMalus và vùng có p < 5% nhưng không cải
thiện dự báo ngoài mẫu.

**5. Phí cơ sở**

Phí cơ sở = tần suất cơ sở 5,24% × mức bồi thường cơ sở 1.690 € × hệ số điều chỉnh 1,2198 = 108,03 €/năm. Hệ số điều
chỉnh gồm hai phần nhân với nhau:

- 1,0014: để tổng phí khớp với tổng chi phí đã chặn.
- 1,2180: để phân bổ phần tổn thất lớn vượt ngưỡng.

Trong quá trình huấn luyện, script tự đối chiếu phí tính từ `bieu_phi.json` với dự báo GLM trên 3.000 hợp đồng; sai
lệch lớn nhất chỉ khoảng 1e-15.

**6. Đánh giá.** Mô hình được đánh giá trên 20% hợp đồng không dùng khi huấn luyện, bằng ba chỉ số: D² (deviance giải
thích được), hệ số Gini và đường Lorenz (khả năng xếp hạng rủi ro), và bảng 10 nhóm rủi ro so phí dự báo với chi phí
thực tế. Kết quả được so với HistGradientBoosting.

## Hạn chế và hướng phát triển

- Mô hình chưa có tương tác giữa các yếu tố, ví dụ tuổi lái × BonusMalus. Mô hình học máy tốt hơn một phần là nhờ
  học được các tương tác này.
- Mức bồi thường gần như không dự báo được. Có thể thử phân phối khác (log-normal, Tweedie), hoặc mô hình riêng cho
  các vụ lớn.
- Hệ số của các vùng ít dữ liệu (Franche-Comté, Alsace, Corse…) có khoảng tin cậy rộng. Có thể làm trơn bằng lý thuyết
  độ tin cậy (credibility) hoặc gộp vùng lân cận.
- Dữ liệu là của Pháp và tính bằng euro. Muốn áp dụng cho Việt Nam cần dữ liệu và biểu phí trong nước.

## Tài liệu tham khảo

- Noll, A., Salzmann, R. & Wüthrich, M. V. (2018). *Case Study: French Motor Third-Party Liability Claims*. SSRN.
- scikit-learn: *Tweedie regression on insurance claims* (ví dụ dùng cùng bộ dữ liệu).
- Dutang, C. & Charpentier, A. *CASdatasets: Insurance datasets* (gói R).
- Ohlsson, E. & Johansson, B. (2010). *Non-Life Insurance Pricing with Generalized Linear Models*. Springer.

*Kết quả chỉ phục vụ mục đích học tập.*
