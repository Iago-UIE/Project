import os
import random
import numpy as np
import csv
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
import cv2

# Configuración
FONT_PATH = "arial.ttf"
OUTPUT_DIR = "word_dataset"
CSV_PATH = os.path.join(OUTPUT_DIR, "word_labels.csv")
WORDS = ["LIMON", "NARANJA", "CIRUELA", "MANZANA", "PERA"]
NUM_SAMPLES_PER_WORD = 200
BLOCK_WIDTH = 128  # Cada "bloque" horizontal para una letra
IMAGE_HEIGHT = 128

# Aumento de datos
datagen = ImageDataGenerator(
    rotation_range=4,
    zoom_range=0.1,
    shear_range=0.1,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest'
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
dataset_info = []

# Función para recortar la imagen

def crop_image(img):
    # Convertir la imagen a escala de grises
    gray = np.array(img)  # No es necesario usar cv2.cvtColor si la imagen ya está en escala de grises
    
    # Aplicar un umbral para binarizar la imagen (blanco y negro)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)  # Umbral ajustado a 200 para más precisión
    
    # Encontrar los contornos de la imagen
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inicializar las coordenadas de recorte
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = -float('inf'), -float('inf')
    
    # Iterar sobre todos los contornos para encontrar las coordenadas extremas
    for contour in contours:
        for point in contour:
            x, y = point[0]
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    
    # Recortar la imagen según las coordenadas mínimas y máximas
    cropped_img = img.crop((min_x, min_y, max_x, max_y))

    # Mostrar la imagen con el rectángulo para depuración
    img_with_rect = np.array(img)
    cv2.rectangle(img_with_rect, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

    plt.show()

    return cropped_img

# Función para generar las imágenes con las palabras
def generate_word_image(word, index):
    word_length = len(word)
    image_width = BLOCK_WIDTH * word_length  # Ajustar ancho en base al número de letras
    font_size = random.randint(50, 70)
    font = ImageFont.truetype(FONT_PATH, font_size)
    
    img = Image.new("L", (image_width, IMAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(img)
    text_bbox = draw.textbbox((0, 0), word, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x_pos = max(5, min((image_width - text_width) // 2 + random.randint(-5, 5), image_width - text_width - 5))
    y_pos = max(5, min((IMAGE_HEIGHT - text_height) // 2 + random.randint(-5, 5), IMAGE_HEIGHT - text_height - 5))
    draw.text((x_pos, y_pos), word, fill="black", font=font)



    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    aug_img = next(datagen.flow(x, batch_size=1))[0].astype(np.uint8)
    img = array_to_img(aug_img)
    # Recortar la imagen generada
    img = crop_image(img)
    file_name = f"{word}_{index}.png"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    img.save(file_path)

    return file_name, word

# Generar imágenes
print(f"Generando dataset ({NUM_SAMPLES_PER_WORD} muestras por palabra)...")
for word in tqdm(WORDS, desc="Generando palabras"):
    for i in range(NUM_SAMPLES_PER_WORD):
        file_path, label = generate_word_image(word, i)
        dataset_info.append([file_path, label])

# Guardar CSV
with open(CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image", "label"])
    writer.writerows(dataset_info)

print(f"Dataset generado con {len(dataset_info)} imágenes.")

# Ejemplo visual
sample_indices = random.sample(dataset_info, 10)
plt.figure(figsize=(15, 6))
for i, (img_name, label) in enumerate(sample_indices):
    img_path = os.path.join(OUTPUT_DIR, img_name)
    img = Image.open(img_path)
    plt.subplot(2, 5, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(label)
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "muestra_dataset.png"))
print("Muestra guardada como muestra_dataset.png")
