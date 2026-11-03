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

git status
git add .
git commit -m "feat: implement helmet detection"
git push

**Lưu ý:** Không commit dữ liệu thực tế chứa biển số, khuôn mặt hoặc video cá nhân lên repository, kể cả khi repository đang Private.

| Prefix        | Mục đích                       |
| ------------- | --------------------------------- |
| `feat:`     | Thêm tính năng                 |
| `fix:`      | Sửa lỗi                         |
| `refactor:` | Tái cấu trúc code              |
| `test:`     | Thêm hoặc sửa test             |
| `docs:`     | Cập nhật tài liệu             |
| `chore:`    | Cấu hình, dependencies, tooling |

# Vietnam Traffic Violation Detection System (VTVDS)

A computer vision system for detecting selected traffic violations
from uploaded images and videos.

## MVP Scope

- Motorcycle rider without helmet detection.
- Red-light violation detection.
- Vietnamese license plate detection and OCR.
- Compare detected plate with user-provided plate.
- Annotated image/video output.
- Human-review status for uncertain results.

## Tech Stack

- Python 3.12
- uv
- OpenCV
- PyTorch (subject to model selection)
- FastAPI
- Gradio
- SQLite
- Git and GitHub

## Development Setup

### 1. Install dependencies

Install uv, then run:

```powershell
uv sync
```

### 2. Activate environment (PowerShell)

.\.venv\Scripts\Activate.ps1

### 3. Run tests

uv run pytest

### 4. Run lint

uv run ruff check .
