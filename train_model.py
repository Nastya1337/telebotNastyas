import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

# Конфигурация
DATA_DIR = 'dataset/ChinaCar'  # Путь к вашему датасету
MODEL_PATH = 'car_model.pth'  # Куда сохранить веса модели
NUM_EPOCHS = 10  # Количество эпох обучения
BATCH_SIZE = 32  # Размер батча
LEARNING_RATE = 0.001  # Скорость обучения

# Список целевых моделей (7 моделей)
TARGET_MODELS = {
    'BYD': ['BYDDenzaZ9GT', 'BYDSeal05DM-i', 'BYDSeal07DM-i'],
    'Changan': ['ChanganShenlanL07'],
    'Chery': ['CheryTiggo7ProMax', 'CheryTiggo9C-DM'],
    'Jetour': ['JetourShanhaiL7']
}

# Проверяем доступность GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")


def prepare_data():
    # Трансформации для данных
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    def filter_and_copy_images(src_dir, dst_dir):
        """Копирует только изображения целевых моделей"""
        dst_dir.mkdir(parents=True, exist_ok=True)

        for brand in TARGET_MODELS:
            brand_path = src_dir / brand
            if not brand_path.exists():
                continue

            for model in TARGET_MODELS[brand]:
                model_path = brand_path / model
                if not model_path.exists():
                    continue

                # Создаем папку класса (модель)
                class_dir = dst_dir / model
                class_dir.mkdir(exist_ok=True)

                # Копируем изображения
                for img_path in model_path.glob('*.*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        shutil.copy(img_path, class_dir / img_path.name)

    # Проверяем, есть ли уже разделение на train и val
    train_dir = Path(DATA_DIR) / 'train'
    val_dir = Path(DATA_DIR) / 'val'

    if train_dir.exists() and val_dir.exists():
        # Создаем временные папки только с целевыми моделями
        temp_train_dir = Path('temp_train')
        temp_val_dir = Path('temp_val')

        if temp_train_dir.exists():
            shutil.rmtree(temp_train_dir)
        if temp_val_dir.exists():
            shutil.rmtree(temp_val_dir)

        filter_and_copy_images(train_dir, temp_train_dir)
        filter_and_copy_images(val_dir, temp_val_dir)

        image_datasets = {
            'train': datasets.ImageFolder(temp_train_dir, data_transforms['train']),
            'val': datasets.ImageFolder(temp_val_dir, data_transforms['val'])
        }
    else:
        # Если нет разделения, создаем его автоматически
        print("Автоматическое разделение датасета на train/val")

        # Создаем временную папку со всеми целевыми моделями
        temp_dir = Path('temp_all')
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        filter_and_copy_images(Path(DATA_DIR), temp_dir)
        full_dataset = datasets.ImageFolder(temp_dir)

        # Разделяем на train и val (80/20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        # Применяем трансформации
        from copy import deepcopy
        train_dataset.dataset = deepcopy(full_dataset)
        val_dataset.dataset = deepcopy(full_dataset)
        train_dataset.dataset.transform = data_transforms['train']
        val_dataset.dataset.transform = data_transforms['val']

        image_datasets = {
            'train': train_dataset,
            'val': val_dataset
        }

    # Получаем список классов (должно быть 7)
    class_names = image_datasets['train'].dataset.classes if 'train' not in image_datasets else image_datasets[
        'train'].classes

    # Создаем загрузчики данных
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }

    print(f"Найдены классы: {class_names}")
    print(f"Количество классов: {len(class_names)}")
    print(f"Размер обучающего набора: {len(image_datasets['train'])}")
    print(f"Размер валидационного набора: {len(image_datasets['val'])}")

    return dataloaders, class_names


# 2. Создание модели
def create_model(num_classes):
    # Загружаем предобученную ResNet18 с новым API
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Заменяем последний слой
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)

    # Переносим модель на GPU (если доступно)
    model = model.to(device)

    return model


# 3. Обучение модели
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Каждая эпоха имеет фазы обучения и валидации
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Режим обучения
            else:
                model.eval()  # Режим оценки

            running_loss = 0.0
            running_corrects = 0

            # Итерация по данным
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Обнуляем градиенты
                optimizer.zero_grad()

                # Прямой проход
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Обратный проход + оптимизация только в фазе обучения
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Статистика
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Сохраняем модель, если точность на валидации улучшилась
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), MODEL_PATH)
                print(f'Модель сохранена с точностью: {best_acc:.4f}')

    return model


# 4. Основная функция
def main():
    # Подготовка данных
    dataloaders, class_names = prepare_data()

    # Создание модели
    model = create_model(len(class_names))

    # Функция потерь и оптимизатор
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Обучение модели
    model = train_model(model, dataloaders, criterion, optimizer, NUM_EPOCHS)

    print('Обучение завершено!')


if __name__ == '__main__':
    main()