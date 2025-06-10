import os

def rename_images_in_folder(folder_path, prefix="img", ext=".jpg", zero_padding=3):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(ext)]
    files.sort()

    # Bước 1: Đổi sang tên tạm để tránh đụng nhau
    temp_names = []
    for idx, filename in enumerate(files):
        old_path = os.path.join(folder_path, filename)
        temp_name = f"temp_{idx}{ext}"
        temp_path = os.path.join(folder_path, temp_name)
        os.rename(old_path, temp_path)
        temp_names.append(temp_name)

    # Bước 2: Đổi từ tên tạm sang tên chuẩn
    for idx, temp_name in enumerate(temp_names, start=1):
        temp_path = os.path.join(folder_path, temp_name)
        new_name = f"{prefix}{str(idx).zfill(zero_padding)}{ext}"
        new_path = os.path.join(folder_path, new_name)
        os.rename(temp_path, new_path)
        print(f"Renamed: {temp_name} -> {new_name}")

# ===== DÙNG =====
rename_images_in_folder("images", prefix="img", ext=".jpg")
