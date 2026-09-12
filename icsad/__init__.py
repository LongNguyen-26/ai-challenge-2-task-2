"""ICS Intrusion Detection - AI Challenge 2 / Task 2.

Các module chính:
    data        : nạp dữ liệu HAI-format từ zip hoặc thư mục, chuẩn hoá min-max
    etapr       : cài đặt lại metric eTaPR (khớp với bản gốc của tác giả)
    features    : sinh ma trận đặc trưng (giá trị + EWMA xuôi/ngược) theo khối
    relation    : mô hình quan hệ tuyến tính (Ridge) - chạy tốt trên CPU
    tcn         : mô hình TCN masked-reconstruction (PyTorch, CPU/GPU)
    scoring     : chuẩn hoá residual -> điểm bất thường
    postprocess : làm trơn, ngưỡng trễ (hysteresis), ghép/lọc đoạn
    synth       : sinh tấn công giả lập để tinh chỉnh ngưỡng khi không có nhãn
    tune        : tìm siêu tham số hậu xử lý theo eTaPR
"""

__version__ = "0.1.0"
