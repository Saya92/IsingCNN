import torch.nn as nn
import torch
class PhaseTransitionClassifier(nn.Module):
    def __init__(self, input_channels:int, conv_layers:list, fc_layers:list, output_size:int, input_height:int, input_width:int, dropout_rate:float=0.2):
        """

        Args:
            input_channels (int): Numero di canali di input.
            conv_layers (list): List of dictionaries to define convolutional layers.
            fc_layers (list): Integer list to define fully connected layers.
            output_size (int): Output dimension.
            dropout_rate (float): Dropout Probability rate
        """
        self.input_channels = input_channels
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.output_size = output_size
        self.dropout= dropout_rate
        self.input_height = input_height
        self.input_width = input_width

        
        super(PhaseTransitionClassifier, self).__init__()

        self.conv_layers = nn.ModuleList()
        in_channels = self.input_channels

        for layer_params in self.conv_layers:
            conv = nn.Conv2d(in_channels, **layer_params)
            self.conv_layers.append(conv)
            self.conv_layers.append(nn.BatchNorm2d(layer_params['out_channels']))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(2, 2))  # Max pooling fisso

            in_channels = layer_params['out_channels']

        self.flatten = nn.Flatten()

        fc_layer_list = []
        in_features = self._calculate_flattened_size(input_channels, conv_layers)

        for out_features in fc_layers:
            fc_layer_list.append(nn.Linear(in_features, out_features))
            fc_layer_list.append(nn.ReLU())
            fc_layer_list.append(nn.Dropout(dropout_rate))
            in_features = out_features

        fc_layer_list.append(nn.Linear(in_features, output_size))
        self.fc_layers = nn.Sequential(*fc_layer_list)

    def _calculate_flattened_size(self)->torch.tensor[int]:
        """This function computes the size of the flattened fully connected layers

        Returns:
            torch.tensor[int]: returns the size of the flattened fully connected layer as a torch tensor
        """
        
        
        with torch.no_grad():
            x = torch.zeros(1, self.input_channels,self.input_height , self.input_width) 
            for layer in self.conv_layers:
                x = layer(x)
            return x.view(1, -1).size(1)

    def forward(self, x:torch.tensor)->torch.tensor:
        """This function computes the forward pass of a neural network

        Args:
            x torch.tensor: input feature of a neural network

        Returns:
            torch.tensor: it returns the output of every layer
        """
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x