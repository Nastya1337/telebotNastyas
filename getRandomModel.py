import random
import os
from glob import glob


def get_random_item(path, is_folder=True, image_extensions=('jpg', 'jpeg', 'png', 'gif', 'bmp')):
    """Получить случайную папку или изображение в указанном пути"""
    if is_folder:
        items = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
    else:
        # Ищем файлы с подходящими расширениями (регистронезависимо)
        items = []
        for ext in image_extensions:
            items.extend(glob(os.path.join(path, f'*.{ext}'), recursive=False))
            items.extend(glob(os.path.join(path, f'*.{ext.upper()}'), recursive=False))

    if not items:
        return None
    return random.choice(items)

def deep_random_path(start_path='./ChinaCar', depth=2):
    """Рекурсивно выбираем случайные папки, затем случайное изображение"""
    current_path = start_path

    # Выбираем случайные папки на указанную глубину
    for _ in range(depth):
        random_folder = get_random_item(current_path, is_folder=True)
        if random_folder is None:
            print(f"Не найдено папок в: {current_path}")
            return None
        current_path = os.path.join(current_path, random_folder)
        print(f"Выбрана папка: {current_path}")


    character = f'{current_path}/character.txt'

    # Выбираем случайное изображение в конечной папке
    random_image = get_random_item(current_path, is_folder=False)
    if random_image is None:
        print(f"Не найдено изображений в: {current_path}")
        return None

    return random_image, character

result = deep_random_path()

if result:
    print(f"\nСлучайное изображение: {result}")
else:
    print("Не удалось найти подходящий файл.")






