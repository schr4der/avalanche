import torch
import matplotlib.pyplot as plt
from dataset import GeoTiffSegmentationDataset
from simpleModel import SimpleFCN  # or UNet if you're using that
from torch.utils.data import random_split

def visualize_prediction():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    generator = torch.Generator().manual_seed(421)
    dataset = GeoTiffSegmentationDataset(3, 3, "../data/swiss_topo_v1/swiss_topo/", "../data/ava_outlines/outlines2018.shp")

    # Create test split
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=generator)
    test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5], generator=generator)

    # Load model and checkpoint
    model = SimpleFCN(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load('my_checkpoint.pth', map_location=device))
    model.eval()

    # Pick one sample from the test dataset
    img, true_mask = test_dataset[10]
    train_img, train_mask = train_dataset[19]
    print("Mask min/max:", true_mask.min(), true_mask.max())
    train_img = train_img.unsqueeze(0).float().to(device)  # add batch dimension

    with torch.no_grad():
        pred_mask = model(train_img)
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask.squeeze().cpu().numpy()
    
    train_img = train_img.squeeze().cpu().numpy()
    train_mask = train_mask.numpy()

    plt.imshow(pred_mask.squeeze(), cmap='viridis')
    plt.title("Raw Sigmoid Output")
    plt.colorbar()  
    # Plot input image, ground truth, and prediction
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(train_img, cmap='gray')
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(train_mask, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask > 0.5, cmap='gray')  # threshold the output
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_prediction()