import torch
from constant import device

def evaluate_model(model, test_loader)->:
    """
    This function allows to evaluate the model on test data. 
    It returns the (mean) accuracy of the model

    Args:
        model (nn.Module): model to evaluate.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Il dispositivo su cui eseguire la valutazione.

    Returns:
        float: L'accuratezza del modello sui dati di test.
    """

    model.eval()  
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  
            outputs = model(inputs)  
            _, predicted = torch.max(outputs, 1)  

            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_predictions
    return accuracy