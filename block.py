import torch
import torch.nn as nn
from attention import Attention
from mlp import MLP



class Block(nn.Module):
    """
    Transformer block 

    Parameters
    ----------
    dim : int 
        Input token dimension 
    
    n_head : int 
        Number of attention heads
    
    mlp_ration : float 
        Determines the number of hidden units in MLP module with respect to dim 
    
    qkv_bias : bool 
        If true we include bias to quuery , key and value 
    p , attn_prob : float
        Dropout probability

    Attributes
    ----------
    norm1 , norm2 : nn.LayerNorm
    attn : Attention Module
    mlp : MLP modules

    """
    def __init__(self , dim , n_head , mlp_ratio =4.0 , qkv_bias = True ,p = 0. , attn_p = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim ,eps = 1e-6)
        self.attn = Attention(dim , n_head , qkv_bias, attn_p , p)
        self.norm2 = nn.LayerNorm(dim , eps = 1e-6)
        self.mlp = MLP(dim , int(dim * mlp_ratio) , dim  , p)

    def forward(self , x):
        """
        Forward pass of transformer block 

        Parameters
        ----------
        x : torch.Tensor
            Shape =  (n_samples , n_patches + 1 , dim)
        
        Returns 
        -------
        torch.Tensor
            Shape = (n_samples, n_patches +1 , dim)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x



