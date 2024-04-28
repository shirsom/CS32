import torch
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from CXRDataset import COVID19_Radiography
from config import * 
from TrainAndTest import TrainAndTest
from DatasetManager import DatasetManager
import numpy as np
import os
import argparse
from utils.file_operations import is_valid_image_file


def create_alexnet(num_classes):
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

    # Modify the last fully connected layer to change the number of output channel
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model

def is_model_trained(model_path=MODEL_PATH):
    return os.path.exists(MODEL_PATH)

def load_model(model_path=MODEL_PATH):
    model = create_alexnet(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def train_model(model_path=MODEL_PATH):
    print("Training Model...")
    model = create_alexnet(num_classes=NUM_CLASSES)

    covid_dataset = COVID19_Radiography()
    
    train_size = int(0.9 * len(covid_dataset))
    test_size = len(covid_dataset) - train_size
    training_dataset, testing_dataset = torch.utils.data.random_split(covid_dataset, [train_size, test_size])
    
    datasets = DatasetManager(training_dataset, testing_dataset)
    t = TrainAndTest(model, datasets, model_path, lr=BASE_LR)

    report = t.train()
    return model


def process_image(image_path):
    if is_valid_image_file(image_path):
        image = Image.open(image_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        input_tensor = preprocess(image)
        return input_tensor

def process_folder(folder_path):
    input_batch = {}
    print("Processing folder:", folder_path)
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            image_path = os.path.join(root, file_name)
            input_batch[image_path] = process_image(image_path)
    
    return {k: v for k, v in input_batch.items() if v is not None}


def is_in_distribution(softmax_output, threshold=0.7):
    max_prob, _ = torch.max(softmax_output, dim=1)
    return max_prob >= threshold

def get_prediction(model, input_path, skip_non_cxr=True):
    if os.path.isfile(input_path) and is_valid_image_file(input_path):
        input_tensor = process_image(input_path)
        input_batch = {input_path: input_tensor}
    elif os.path.isdir(input_path):
        input_batch = process_folder(input_path)
    else:
        raise ValueError('Invalid input path: Unsupported file extension')
    
    results = ""
    for path, image_tensor in input_batch.items():
        image_input = image_tensor.unsqueeze(0)  # Add a batch dimension
        output = model(image_input)
        softmax_probs = F.softmax(output, dim=1)

        if not is_in_distribution(softmax_probs):
            if not skip_non_cxr:
                raise ValueError('The uploaded image is not a chest x-ray scan')
            else:
                results += f'{path}: The uploaded image is not a chest x-ray scan\n'
                continue

        probabilities_np = softmax_probs.detach().numpy()
        probabilities_percent = [[round(prob * 100, 1) for prob in class_probs] for class_probs in probabilities_np]

        _, predicted_idx = torch.max(output, 1)
        predicted_label = predicted_idx.item()

        prediction = "Abnormal" if predicted_label else "Normal"
        results += f"{path}: {prediction} with probability {probabilities_percent[0][predicted_label]}%\n"
    
    return results 
    
def get_model():
    if is_model_trained():
        return load_model()
    return train_model()

def run(image_path, skip_non_cxr=True, output_path=False):
    model = get_model()
    prediction = get_prediction(model, image_path, skip_non_cxr)
    if output_path:
        with open(output_path, 'a') as file:
            file.write(str(prediction))
        print(f"Results have been saved to {output_path}")
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to either a single x-ray to screen or a folder of x-ray scans")
    parser.add_argument("--output-path", help="Optional file path to save the output to", type=str)
   
    args = parser.parse_args()
    print(run(args.input_path, output_path=args.output_path))