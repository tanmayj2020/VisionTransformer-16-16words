import torch 
import torch.nn as nn


class MLP(nn.Module):
    """
    NN before attention block 

    Parameters 
    ----------
    input_dim : int 
        Number of input features

    output_dim : int 
        Number of output features 
    
    hidden_dim : int 
        Number of hidden dimensions
    
    p : float 
        Dropout probability 

    Attribute
    ---------
    fc : nn.Linear 
        First linear layer

    act : nn.GELU 
        GELU activation function 

    fc2 : nn.Linear 
        The second linear layer
    
    drop : nn.Dropout 
        Dropout Layer

    """

    def __init__(self , input_dim , hidden_dim , output_dim , p):
        super().__init__()
        self.fc1 = nn.Linear(input_dim , hidden_dim )
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim , output_dim)
        self.drop = nn.Dropout(p)

    def forward(self , x):
        """
        Forward of multi layer perceptron

        Parameters 
        ----------
        x: torch.Tensor
            Shape = (n_samples , n_patches + 1 ,input_dim)

        Returns 
        -------
        torch.Tensor 
            Shape = (n_samples , n_patches + 1 ,output_dim)       
        """
        x = self.fc1(x) #(n_samples , n_patches + 1,hidden_dim)
        x = self.act1(x) #(n_samples , n_patches + 1, hidden_dim)
        x = self.fc2(x) # (n_samples , n_patches + 1 , output_dim)
        x = self.drop(x) #(n_samples , n_patches + 1, output_dim)
        return x

    