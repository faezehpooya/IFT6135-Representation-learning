import torch
from typing import List, Tuple
from torch import nn
import sys

class Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
       
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
    
    def forward(self, input):
        """
            :param input: [bsz, in_features]
            :return result [bsz, out_features]
        """
        # output = self.linear(input)
        _, y = input.shape
        if y != self.in_features:
            sys.exit(f'Wrong Input Features. Please use tensor with {self.in_features} Input Features')
        output = input @ self.weight.t() + self.bias
        return output


class MLP(torch.nn.Module):
    # 20 points
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, activation: str = "relu"):
        super().__init__() 
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        assert len(hidden_sizes) > 1, "You should at least have one hidden layer"
        self.num_classes = num_classes
        self.activation = activation
        assert activation in ['tanh', 'relu', 'sigmoid'], "Invalid choice of activation"
        self.hidden_layers, self.output_layer = self._build_layers(input_size, hidden_sizes, num_classes)
        
        # Initializaton
        self._initialize_linear_layer(self.output_layer)
        for layer in self.hidden_layers:
            self._initialize_linear_layer(layer)
    
    def _build_layers(self, input_size: int, 
                        hidden_sizes: List[int], 
                        num_classes: int) -> Tuple[nn.ModuleList, nn.Module]:
        """
        Build the layers for MLP. Be ware of handlling corner cases.
        :param input_size: An int
        :param hidden_sizes: A list of ints. E.g., for [32, 32] means two hidden layers with 32 each.
        :param num_classes: An int
        :Return:
            hidden_layers: nn.ModuleList. Within the list, each item has type nn.Module
            output_layer: nn.Module
        """
        architecture = nn.ModuleList([Linear(input_size, hidden_sizes[0])])
        for i in range(1, len(hidden_sizes)):
            architecture.append(Linear(hidden_sizes[i-1], hidden_sizes[i]))
        # output
        return architecture, Linear(hidden_sizes[-1], num_classes)
    
    def activation_fn(self, activation, inputs: torch.Tensor) -> torch.Tensor:
        """ process the inputs through different non-linearity function according to activation name """
        if activation=="tanh":
          return torch.nn.Tanh()(inputs)
        if activation=="relu":
          return torch.nn.ReLU()(inputs)
        if activation =="sigmoid":
          return torch.nn.Sigmoid()(inputs)

        
    def _initialize_linear_layer(self, module: nn.Linear) -> None:
        """ For bias set to zeros. For weights set to w """
        torch.nn.init.xavier_normal_(module.weight)
        torch.nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """ Forward images and compute logits.
        1. The images are first fattened to vectors. 
        2. Forward the result to each layer in the self.hidden_layer with activation_fn
        3. Finally forward the result to the output_layer.
        
        :param images: [batch, channels, width, height]
        :return logits: [batch, num_classes]
        """
        x = images.view(images.shape[0], -1)  # Step 1
        for layer in self.hidden_layers:      # Step 2
            x = layer(x)
            x = self.activation_fn(self.activation, x)
        logits = self.output_layer(x)         # Step 3
        return logits

