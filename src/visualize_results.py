import torch
import matplotlib.pyplot as plt
from dataset import GeoTiffSegmentationDataset
from mini_CNN import SimpleFCN  # or UNet if you're using that
from uNetSmaller import UNet_Modified
from UNet3 import UNet3
from model import UNet
from torch.utils.data import random_split

# Utility script to visualize model predictions alongside computed data features
def visualize_prediction():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    generator = torch.Generator().manual_seed(421)
    dataset = GeoTiffSegmentationDataset(3, 3, "../data/swiss_topo_v2/swiss_topo/", "../data/ava_outlines/outlines2018.shp")

    # Create test split
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=generator)
    test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5], generator=generator)

    # Load model and checkpoint
    model = UNet(in_channels=4, num_classes=1).to(device)
    model.load_state_dict(torch.load('my_checkpoint.pth', map_location=device))
    model.eval()

    # Pick one sample from the test dataset
    img_full, true_mask = test_dataset[7]
    img = img_full[0]
    print(f"Max: {torch.max(img).item()}, Min: {torch.min(img).item()}")
    img_slope = img_full[1]
    print(f"Max: {torch.max(img_slope).item()}, Min: {torch.min(img_slope).item()}")
    img_aspect = img_full[2]
    print(f"Max: {torch.max(img_aspect).item()}, Min: {torch.min(img_aspect).item()}")
    img_curve = img_full[3]
    print(f"Max: {torch.max(img_curve).item()}, Min: {torch.min(img_curve).item()}")
    img_slope = img_slope.cpu().numpy()
    img_aspect = img_aspect.cpu().numpy()
    img = img.cpu().numpy()
    img_curve = img_curve.cpu().numpy()
    # train_img, train_mask = train_dataset[19]
    # img = img.unsqueeze(0).float().to(device)  # add batch dimension
    img_full = img_full.unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred_mask = model(img_full)
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask.squeeze().cpu().numpy()
    # img_slope = img_full[1]
    true_mask = true_mask.numpy()

    plt.imshow(pred_mask.squeeze(), cmap='viridis')
    plt.title("Raw Sigmoid Output")
    plt.colorbar()  
    # Plot input image, ground truth, and prediction
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 6, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Input Image")
    plt.axis('off')
    plt.subplot(1, 6, 2)
    plt.imshow(img_slope, cmap='gray')
    plt.title("Gradient")
    plt.axis('off')
    plt.subplot(1, 6, 3)
    plt.imshow(img_curve, cmap='gray')
    plt.title("Aspect")
    plt.axis('off')
    plt.subplot(1, 6, 4)
    plt.imshow(img_aspect, cmap='gray')
    plt.title("Curvature")
    plt.axis('off')

    plt.subplot(1, 6, 5)
    plt.imshow(true_mask, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis('off')

    print(pred_mask)
    # booled = (pred_mask>0.1).astype(int)
    # booled [0][0] = 0
    # print(booled)
    plt.subplot(1, 6, 6)
    plt.imshow(pred_mask>0.5, cmap='gray')  # threshold the output
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_prediction()