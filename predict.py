import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import requests

from dataset import SketchDataset, get_transforms
from model import SketchCNN

def make_payload(predictions):
    """predictions: list of 1000 integer label IDs for the test set."""
    return {"solution": predictions}

def post_answer(payload, api_key):
    url = "https://competitions.aiolympiad.my/api/maio2026_qualis/maio2026_sketch_classification"
    response = requests.post(url=url, json=payload, headers={"X-API-Key": api_key})
    if response.status_code == 200:
        return response.json()
    else:
        return (
            f"Failed to submit, status code is {response.status_code}\n{response.text}"
        )

def predict_test_set(data_dir, model_path, use_api=False, api_key=None, device="cpu"):
    _, val_test_transform = get_transforms()
    
    # Load test dataset which returns (image, img_name)
    test_dataset = SketchDataset(data_dir, split="test", transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Init Model and Load Best Weights
    model = SketchCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    
    print("Generating predictions for 1000 test images...")
    with torch.no_grad():
        for images, img_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            
    print(f"Generated {len(all_preds)} predictions.")
    
    if len(all_preds) != 1000:
        print(f"WARNING: Expected 1000 predictions, but got {len(all_preds)}!")
        
    print("Top 10 Predictions:", all_preds[:10])
    
    if use_api:
        if not api_key:
            api_key = input("Please enter your API Key: ")
        print("Submitting to API...")
        payload = make_payload(all_preds)
        result = post_answer(payload, api_key)
        print("Submission Result:")
        print(result)
        
    return all_preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--submit", action="store_true", help="Submit predictions to API")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predict_test_set(data_dir="sketch_clf", model_path=args.model_path, use_api=args.submit, device=device)
