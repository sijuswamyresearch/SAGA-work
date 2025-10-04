# SAGA-work
This repository contains supplementary documents and Python code for the work

## Index
- [SAGA – block diagram](#saga--block-diagram)
- [Python code for SAGA](#python-code-for-saga)
- [Supplementary documents](#supplementary-documents)
- [Models](#models)

## SAGA – block diagram


The block diagram of SAGA is shown below:

![SAGA](https://github.com/sijuswamyresearch/SAGA-work/blob/main/Supporting%20documents/SAGA.png)

## Python code for SAGA

```python

class SAGA(nn.Module):
    """
    This provides more nuanced, spatially-aware control on AF.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # A depthwise convolution to capture spatial context.
        # This is shared between the boost and the gate.
        self.spatial_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.spatial_bn = nn.BatchNorm2d(channels)
        
        # A lightweight 1x1 convolution to create the gate from the context.
        # This will learn how to transform the context T_x into an optimal gate.
        self.gate_generator = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)

        # Initialization
        nn.init.kaiming_normal_(self.spatial_conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.spatial_bn.weight, 1)
        nn.init.constant_(self.spatial_bn.bias, 0)
        
        # Initialize the gate generator so the initial gate is ~1.0.
        # A positive bias on a sigmoid makes its output start close to 1.

        nn.init.constant_(self.gate_generator.weight, 0)
        nn.init.constant_(self.gate_generator.bias, 2.0) # sigmoid(2.0) ˜ 0.88

    def forward(self, x):
        # 1. Generate the shared spatial context
        T_x = self.spatial_bn(self.spatial_conv(x))
        
        # 2. Calculate the standard SAGA residual boost
        boost = F.relu(T_x - x)
        
        # 3. Generate a dynamic, per-pixel gate from the context
        # The gate will learn where to apply the boost strongly vs. weakly.
        gate = torch.sigmoid(self.gate_generator(T_x))
        
        # 4. Apply the gate to the boost and add to the original input
        
        output = x + (gate * boost)
        
        return output

```


>*SAGA code explanation:*
This class defines the SAGA (Spatially-Aware Gated Activation) module, which provides nuanced, spatially-aware control on activation functions in neural networks. The module uses depthwise convolutions to capture spatial context and generates a dynamic gate to modulate the activation boost based on this context. The implementation includes initialization of weights and biases to ensure effective learning.

### Supplementary documents
The supplementary documents for this work can be found in the `Supporting documents` folder.

[Supporting documents](.\Supporting documents)

They include:

- Comparison of training and validation loss of various AFs.
- Sample deblurring results on the CT dataset
- Sample deblurring results on the Osteoporosis Xray dataset

## Models

Sample trained models can be found in the `models` folder.
They include:

- A trained DeblurResNet model with SAGA AF on the CT dataset
- A trained DeblurResNet model with SAGA AF on the Osteoporosis Xray dataset

[Models](.\models)


>*Complete code:*
The complete code for training and evaluating models with SAGA wiil be uploaded upon publication of the work.
