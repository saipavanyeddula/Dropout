import torch
import numpy as np
from models.common import DetectMultiBackend # Replace this with your local YOLOv8 model
from pathlib import Path
from utils.dataloaders import LoadImages

# Load the pre-trained model (with dropout added in your custom YOLOv8)
model = DetectMultiBackend("C:/Users/avari/PycharmProjects/onlydropout/yolov5/yolov5m.pt")

# Function to enable dropout during testing
def enable_dropout(model):
    """ Function to activate dropout layers during testing """
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

# MC Dropout prediction function
def mc_dropout_predict(model, image, n_passes=10):
    """ Function to perform multiple forward passes with MC Dropout """
    model.eval()  # Ensure the model is in eval mode (but dropout stays active)
    enable_dropout(model)  # Force dropout to stay active

    predictions = []

    # Run multiple forward passes
    for _ in range(n_passes):
        with torch.no_grad():
            result = model(image)
            print(result)
            predictions.append(result)

    # Stack predictions to compute uncertainty (variance)
    predictions = torch.stack(predictions, dim=0)
    mean_prediction = torch.mean(predictions, dim=0)
    uncertainty = torch.var(predictions, dim=0)

    return mean_prediction, uncertainty

# Test function that applies MC Dropout
def test_with_mc_dropout(model, test_images_dir, n_passes=10):
    """ Perform testing on all images with MC Dropout """
    image_paths = list(Path(test_images_dir).glob("*.jpg"))
    uncertain_predictions = []

    for image_path in image_paths:
        image = LoadImages(image_path)  # Implement image loading function
        print("image loaded")
        mean_pred, uncertainty = mc_dropout_predict(model, image, n_passes=n_passes)
        print(mean_pred)
        # Store image path, predictions, and uncertainty
        uncertain_predictions.append((image_path, mean_pred, uncertainty))

    # Sort by uncertainty and store top 10 uncertain predictions
    uncertain_predictions = sorted(uncertain_predictions, key=lambda x: torch.max(x[2]), reverse=True)
    top_uncertain_predictions = uncertain_predictions[:10]

    # Output results for further human annotation
    for idx, (img_path, pred, uncert) in enumerate(top_uncertain_predictions):
        print(f"Image {idx+1}: {img_path} - Max Uncertainty: {torch.max(uncert)}")

    return top_uncertain_predictions

# Path to the test images
test_images_dir = "C:/Users/avari/PycharmProjects/onlydropout/yolov5/data/images/bus.jpg"

# Run testing with MC Dropout
test_with_mc_dropout(model, test_images_dir, n_passes=4)
