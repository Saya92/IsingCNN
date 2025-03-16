# IsingCNN
 # 2D Ising Model Simulation and Phase Transition Classification with CNN

This project simulates the 2D Ising model using the Metropolis algorithm and classifies phase transitions using a Convolutional Neural Network (CNN).

## Description

The 2D Ising model is a fundamental model in statistical physics used to study phase transitions, particularly ferromagnetism. This project implements the Metropolis algorithm to simulate the Ising model and uses a CNN to classify the phases (ordered/disordered) based on the spin configurations.

## Files

-   `src/constant.py`: Defines constant values and device settings.
-   `src/dataset.py`: Generates the Ising dataset, performs train-test split, and prepares it for training (converts data to DataLoader and TensorDataset).
-   `src/IsingGrid.py`: Implements the 2D Ising grid and Metropolis algorithm.
-   `src/model_evaluator.py`: Evaluates the trained CNN model.
-   `src/model_trainer.py`: Trains the CNN model.
-   `src/model.py`: Defines the CNN model architecture.
-   `src/transform.py`: Defines data transformations for the input data.
-   `test3.ipynb`: Jupyter notebook for testing and experimentation.
-   `data/`: Directory to store the generated dataset.

## Dependencies

-   Python 3.x
-   PyTorch (`torch`)
-   NumPy (`numpy`)
-   Scikit-learn (`sklearn`)
-   Matplotlib (`matplotlib`)

You can install these dependencies using pip:

```bash
pip install torch numpy scikit-learn matplotlib