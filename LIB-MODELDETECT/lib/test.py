import torch
import paddle
import easyocr

#Torch
print("Torch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Torch CUDA version:", torch.version.cuda)
    print("Torch GPU device name:", torch.cuda.get_device_name(0))

#Paddle
print("Paddle compiled with CUDA:", paddle.device.is_compiled_with_cuda())
if paddle.device.is_compiled_with_cuda():
    print("Paddle CUDA version:", paddle.version.cuda())
    print("Paddle GPU count:", paddle.device.get_device_count())
    print("Paddle current device:", paddle.get_device())

#EasyOCR
try:
    reader = easyocr.Reader(['en'], gpu=True)  # gpu=True để kích hoạt GPU
    print("EasyOCR using device:", reader.device)  # 'cuda:0' nếu dùng GPU
except Exception as e:
    print("EasyOCR GPU check failed:", e)
