import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device:str):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def prepare_data(self, dataset:torch.tensor, labels:torch.tensor, batch_size:int, random_state:int=42)->None:
        """This function performs a simple random train test split and creates training, validation and test dataloader iterables.

        Args:
            dataset (torch.tensor): _description_
            labels (torch.tensor): _description_
            batch_size (int): _description_
            random_state (int, optional): _description_. Defaults to 42.
        """
        
        train_dataset, test_dataset, train_labels, test_labels = train_test_split(
            dataset, labels, test_size=0.2, random_state=random_state
        )
        train_dataset, val_dataset, train_labels, val_labels = train_test_split(
            train_dataset, train_labels, test_size=0.25, random_state=random_state
        )

        train_dataset = torch.utils.data.TensorDataset(train_dataset, train_labels)
        val_dataset = torch.utils.data.TensorDataset(val_dataset, val_labels)
        test_dataset = torch.utils.data.TensorDataset(test_dataset, test_labels)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def train(self, epochs:int)->tuple[list]:
        """This function computes the loop training of a generic neural network

        Args:
            epochs (int): Number of epoch training

        Returns:
            tuple[list]: it returns training losses, validation losses and validation accuracies
        """
        
        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            train_loss /= len(self.train_loader.dataset)
            train_losses.append(train_loss)

            val_loss, val_accuracy = self.validate()
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")

        return train_losses, val_losses, val_accuracies

    def validate(self)-> tuple:
        """This function calculates loss and accuracies using the validation set

        Returns:
            tuple: Returns the average test loss and the test accuracy
        """
        self.model.eval()
        val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(self.val_loader.dataset)
        val_accuracy = correct_predictions / total_predictions
        return avg_val_loss, val_accuracy

   