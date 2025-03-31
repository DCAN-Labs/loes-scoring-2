import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


# Define the logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        # Apply the linear transformation and sigmoid activation
        outputs = torch.sigmoid(self.linear(x))
        return outputs


# Create a simple synthetic dataset
class SyntheticBinaryDataset(Dataset):
    def __init__(self, n_samples=1000, input_dim=2, random_seed=42):
        np.random.seed(random_seed)
        
        # Generate random features
        self.X = np.random.randn(n_samples, input_dim)
        
        # Generate labels based on a linear decision boundary with some noise
        z = np.sum(self.X, axis=1) + np.random.randn(n_samples) * 0.5
        self.y = (z > 0).astype(np.float32).reshape(-1, 1)
        
        # Convert to PyTorch tensors
        self.X = torch.FloatTensor(self.X)
        self.y = torch.FloatTensor(self.y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Training function
def train_logistic_regression(model, train_loader, criterion, optimizer, epochs=100):
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Store average loss for this epoch
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
            
    return losses


# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X)
            predicted = (outputs > 0.5).float()
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy


# Function to visualize the decision boundary (for 2D data)
def plot_decision_boundary(model, X, y):
    model.eval()
    
    # Set min and max values with some margin
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Create a meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Convert to PyTorch tensors and make predictions
    mesh_input = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    with torch.no_grad():
        mesh_output = model(mesh_input).numpy().reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, mesh_output > 0.5, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.savefig('logistic_regression_boundary.png')
    plt.show()


# Main function to run the entire pipeline
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Hyperparameters
    input_dim = 2
    batch_size = 32
    learning_rate = 0.01
    epochs = 100
    
    # Create dataset and split into train/test
    full_dataset = SyntheticBinaryDataset(n_samples=1000, input_dim=input_dim)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = LogisticRegression(input_dim)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Train the model
    losses = train_logistic_regression(model, train_loader, criterion, optimizer, epochs)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    
    # Evaluate the model
    accuracy = evaluate_model(model, test_loader)
    
    # Plot decision boundary
    X_data = full_dataset.X
    y_data = full_dataset.y
    plot_decision_boundary(model, X_data, y_data)
    
    # Print model parameters
    print('Model Parameters:')
    for name, param in model.named_parameters():
        print(f'{name}: {param.data.numpy()}')


if __name__ == "__main__":
    main()
