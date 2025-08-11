from setuptools import setup, find_packages
from pathlib import Path

# Bạn cần đảm bảo định nghĩa __version__ trước khi dùng
__version__ = "1.0.0"

# Nội dung mô tả dài
root = Path(__file__).parent
long_description = "# MultiDetect Library\nThư viện phát hiện và nhận dạng đa đối tượng (biển số, container, khuôn mặt)."

# Danh sách dependencies đọc từ requirements.txt hoặc khai báo trực tiếp
req_path = root / "requirements.txt"
if req_path.exists():
    install_requires = [
    ln.strip()
    for ln in req_path.read_text(encoding="utf-8").splitlines()
    if ln.strip() and not ln.strip().startswith("#")
    ]
else:
    install_requires = []

setup(
    name="multidetect-library",
    version=__version__,
    author="GitGud",
    author_email="tienteni01@gmail.com",
    description="Thư viện phát hiện và nhận dạng đa đối tượng (biển số, container, khuôn mặt)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multidetect-lib",

    # Tự động tìm các package có __init__.py, loại trừ folders không cần
    packages=find_packages(exclude=("example", "examples", "tests", "docs")),

    # Phiên bản Python hỗ trợ
    python_requires="=3.10",

    # Cài đặt phụ thuộc
    install_requires=install_requires,

    # Tùy chọn thêm
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
    },

    # Đóng gói dữ liệu tĩnh trong package
    include_package_data=True,

    # Khai báo dữ liệu tĩnh cần nhúng trong gói
    # Ở đây nhúng model YOLO (*.pt) và file classifier (*.joblib) trong lib/model
    package_data={
        "lib.model": ["*.pt", "*.joblib"],
    },
  
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
    ],
)