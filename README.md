# Đọc hoá đơn trong một file

Ảnh hoá đơn vào, một dòng sổ chi tiêu ra. Project ráp kiến thức Buổi 20 đến 23 vào một file Python, có giao diện Gradio để thử và bộ eval 30 hoá đơn để chấm.

Không cần key, không gọi dịch vụ trả tiền nào. Mọi model chạy ngay trên máy bạn.

## App làm gì

| Bước | Làm gì | Học ở buổi |
|---|---|---|
| 1 · Cổng vào | Đo độ nét (OpenCV). Ảnh rõ thì hỏi SigLIP 2: "đây có phải ảnh hoá đơn?" | 22, 23 (SigLIP 2 chính là ViT vừa mở hộp) |
| 2 · Đọc chữ | PaddleOCR tìm khung chữ, rồi đọc chữ trong từng khung | 20, 21 |
| 3 · Trích | Lấy tên quán, ngày, tổng tiền bằng luật chịu được lỗi OCR | 21 |
| 4 · Sensor | Hoá đơn tự kiểm: ngày phải thuộc năm nay, các món cộng lại = tạm tính, tạm tính − giảm giá = tổng | 22 |
| 5 · Quyết định | Mọi phép kiểm qua thì **GHI SỔ**, có lỗi thì **CẦN NGƯỜI XEM**, ảnh không dùng được thì **CHẶN** | 22 |

App chạy các phép kiểm rẻ mà chắc chắn trước (đo độ nét chỉ mất vài mili giây), rồi mới gọi model. Khi các con số không khớp, app chuyển cho người xem thay vì đoán.

## Cài đặt

Cần **Python 3.12**. PaddlePaddle chưa có bản cho Python 3.13.

```bash
cd hoa-don-mini
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # 3 đến 5 phút
```

Lần đầu chạy model, máy tải thêm khoảng **1,5 GB** (SigLIP 2 và PaddleOCR). Các lần sau dùng lại, không tải nữa.

Máy chỉ có Python 3.13 trở lên thì cài `uv` để tạo venv 3.12:

```bash
pip install uv
uv venv --python 3.12 venv
source venv/bin/activate
uv pip install -r requirements.txt
```

python --version
git --version
uv --version
nvidia-smi

.\.venv\Scripts\Activate.ps1 - active uv

uv run python --version
uv run python -c "import sys; print(sys.executable)"

git add .
git status
git commit -m "chore: initialize project environment"