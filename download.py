import urllib.request
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def download_file(url, filename):
    """Скачивает файл с обработкой ошибок"""
    try:
        print(f"Скачиваю {filename}...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            with open(filename, 'wb') as out_file:
                out_file.write(response.read())
        print(f"✓ {filename} успешно скачан")
        return True
    except Exception as e:
        print(f"✗ Ошибка при скачивании {filename}: {e}")
        return False

if not os.path.exists('yolo_files'):
    os.makedirs('yolo_files')

files = {
    'coco.names': [
        'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names',
    ],
    'yolov3-tiny.cfg': [
        'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg',
    ],
    'yolov3-tiny.weights': [
        'https://pjreddie.com/media/files/yolov3-tiny.weights',
    ]
}

print("Пытаемся скачать файлы YOLOv3-tiny...")

success_count = 0
for filename, urls in files.items():
    filepath = os.path.join('yolo_files', filename)
    
    if os.path.exists(filepath):
        print(f"✓ {filename} уже существует")
        success_count += 1
        continue
    
    downloaded = False
    for url in urls:
        if download_file(url, filepath):
            downloaded = True
            success_count += 1
            break
    
    if not downloaded:
        print(f"✗ Не удалось скачать {filename}")

if success_count == 3:
    print("\n🎉 Все файлы успешно скачаны!")
    print("Теперь вы можете запустить детектор: python detector.py")
else:
    print(f"\n⚠️ Скачано {success_count} из 3 файлов")
    if success_count > 0:
        print("Часть файлов скачана. Можно попробовать запустить с имеющимися файлами.")