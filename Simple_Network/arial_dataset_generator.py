import os
import random
import string
import numpy as np
import csv
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

##Generación de imágenes a partir de archivos .ttf
from tqdm import tqdm


### Fución de preprocesamiento por defecto de keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img

# Configuración
FONT_PATH = "arial.ttf"
OUTPUT_DIR = "arial_dataset"
CSV_PATH = os.path.join(OUTPUT_DIR, "arial_letters.csv")
NUM_SAMPLES_PER_LETTER = 200
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

# Transformaciones automáticas
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    shear_range=0.1,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest'
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
dataset_info = []

def generate_letter_image(letter, index):
    font_size = random.randint(50, 90)
    font = ImageFont.truetype(FONT_PATH, font_size)
    img = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(img)
    text_bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x_pos = max(5, min((IMAGE_WIDTH - text_width) // 2 + random.randint(-10, 10), IMAGE_WIDTH - text_width - 5))
    y_pos = max(5, min((IMAGE_HEIGHT - text_height) // 2 + random.randint(-10, 10), IMAGE_HEIGHT - text_height - 5))
    draw.text((x_pos, y_pos), letter, fill="black", font=font)

    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    aug_img = next(datagen.flow(x, batch_size=1))[0].astype(np.uint8)
    img = array_to_img(aug_img)

    file_name = f"{letter}_{index}.png"  # solo el nombre
    file_path = os.path.join(OUTPUT_DIR, file_name)  # ruta completa para guardar
    img.save(file_path)

    return file_name, letter  # devolvemos solo el nombre para guardar en el CSV


# Generación del dataset
print(f"Generando dataset ({NUM_SAMPLES_PER_LETTER} muestras por letra)...")
alphabet = string.ascii_uppercase
for letter in tqdm(alphabet, desc="Generando letras"):
    for i in range(NUM_SAMPLES_PER_LETTER):
        file_path, character = generate_letter_image(letter, i)
        dataset_info.append([file_path, character])

# Guardar en CSV
print(f"Guardando CSV en {CSV_PATH}")
with open(CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["titulo", "texto"])
    writer.writerows(dataset_info)

print(f"Dataset completo: {len(dataset_info)} imágenes generadas.")

# Mostrar muestra aleatoria
plt.figure(figsize=(15, 8))
sample_indices = random.sample(range(len(dataset_info)), min(10, len(dataset_info)))
for i, idx in enumerate(sample_indices):
    img_name, letter = dataset_info[idx]
    img_path = os.path.join(OUTPUT_DIR, img_name)
    img = Image.open(img_path)
    plt.subplot(2, 5, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Letra: {letter}")
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"))
print("Muestra guardada como sample_images.png")
