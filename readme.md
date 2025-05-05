# Understanding CTC Loss with Simple Network and Hugging Face OCR

This project demonstrates how **Connectionist Temporal Classification (CTC) Loss** is used in OCR (Optical Character Recognition) with a simple neural network and the `hugging_face_ocr` library. It involves training a model using CTC loss to align input sequences (images) with output sequences (text), particularly when the alignment between the two is not explicitly given.

## Project Structure

- `dataset_generator.py` and `Simple_Network/arial_dataset_generator.py`: Scripts to generate datasets used for training and evaluation.
- `letter_classifier.pth`: Pre-trained weights of a simple neural network model using CTC loss.
- `hugging_face_ocr.py`: Invoice reading hugging_face model
- `CTC_copy.ipynb`: Script to train a basic CRNN model with CTC loss. It uses letter_classifier.pth.
- `Red_Simple.copy.ipynb`: Script that generates the model using letter_classifier.pth.
- Imports are available in every file

## What is CTC Loss?

CTC (Connectionist Temporal Classification) loss is a type of objective function used to train neural networks for sequence prediction problems where the alignment between the inputs and outputs is unknown.

In OCR, the input is typically a sequence of image features extracted by a CNN, and the output is a sequence of characters. CTC allows the model to learn how to map the variable-length input sequence to a shorter output sequence (e.g., the text in an image) without needing exact alignment between them.

### Key Advantages:

- Handles variable-length input and output sequences.
- No need for pre-aligned labeled data.
- Efficient for tasks like OCR, speech recognition, and handwriting recognition.

## How to Run


1. **Remember to doqwnload the necesary libraries**:

2. **Generate the Dataset** (must be done first):
   arial_dataset_generator.py
3. **Get the .pth file**:
    Red_Simple copy.ipynb
4. **get the other dataset**:
    dataset_generator.py
5. **Train the model**:
    CTC copy.ipynb
