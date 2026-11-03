

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
