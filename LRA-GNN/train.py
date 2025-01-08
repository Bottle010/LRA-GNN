import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.lra_gnn import LRA_GNN
from models.progressive_rl import ProgressiveRLAgent
from dataset import AgeEstimationDataset
from utils import calculate_metrics
import matplotlib.pyplot as plt
import os

def train_model(model, rl_agent, train_loader, optimizer, criterion, num_epochs, device, save_path):
   
    model.train()
    losses = []
    maes = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass through LRA-GNN
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels.unsqueeze(1).float())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Evaluate the model
        model.eval()
        with torch.no_grad():
            predictions = model(images)
            metrics = calculate_metrics(predictions, labels)
            mae = metrics['MAE']
            maes.append(mae)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, MAE: {mae:.2f}")
        
        model.train()
        losses.append(running_loss / len(train_loader))
    
    # Visualize loss and MAE
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(maes, label='MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE Over Epochs')
    plt.legend()
    
    plt.show()
    
    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Usage example
if __name__ == "__main__":
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize the dataset
    dataset = AgeEstimationDataset(root_dir='path/to/dataset', transform=transform)
    
    # Create data loader
    batch_size = 32
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the LRA-GNN model
    model = LRA_GNN(num_layers=12, num_heads=8, in_channels=3, hidden_channels=128, out_channels=1)
    model.to(device)
    
    # Initialize the Progressive RL agent
    state_dim = 128  # Example state dimension (output from LRA-GNN)
    action_dim = 5   # Example action dimension (age groups)
    rl_agent = ProgressiveRLAgent(state_dim, action_dim)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Train the model
    num_epochs = 10
    save_path = 'path/to/save/model.pth'
    train_model(model, rl_agent, train_loader, optimizer, criterion, num_epochs, device, save_path)