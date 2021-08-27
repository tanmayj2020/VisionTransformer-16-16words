import torch 
import torch.nn as nn 




class Attention(nn.Module):
    """
    Implements attention 

    Parameters 
    ----------
    dim :int 
        Input dimension of the per token feature
    
    n_head : int 
        Number of attention heads
    
    qkv_bias : bool 
        If True then we include bias to query , key and value projections
    
    atten_p : float 
        Dropout probability of attention matrix 
    
    proj_p : float 
        Dropout probability of the projecttion to output tensor 

    Attributes
    ----------
    scale : float 
        Prevents softmax output low gradients
    qkv : 
        Linear projection of input embedding to query , key and value 
    proj :
        Linear mapping from concateated output of attention to same space

    attn_drop , proj_drop : nn.Dropout 
        Applying dropouts to prevent overfitting 
    """

    def __init__(self , input_dim , n_head = 12 , qkv_bias = True , attn_prob = 0. , proj_prob = 0.):
        super().__init__()
        self.input_dim = input_dim 
        self.n_head = n_head
        self.head_dim = input_dim // n_head
        self.scale = self.head_dim ** (-0.5)
        self.qkv = nn.Linear(input_dim , input_dim * 3 , bias = qkv_bias)
        self.attn_drop = nn.Dropout(p = attn_prob)
        self.proj = nn.Linear(input_dim , input_dim)
        self.proj_drop = nn.Dropout(p = proj_prob)
    

    def forward(self , x):
        """
        Forward pass attention block 

        Parameters
        ----------
        x : torch.Tensor
            Shape = (n_samples,n_patches + 1 , input_dim)
        
        Returns 
        -------
        torch.Tensor
            Shape = (n_samples , n_patches + 1, input_dim)
        
        
        """
        n_samples , n_tokens , input_dim = x.shape
        qkv = self.qkv(x) # (n_samples , n_patches + 1, embed_dim * 3)
        qkv = qkv.reshape(n_samples, n_tokens , 3 , self.n_head , self.head_dim) #(n_samples , n_patches + 1, 3,  n_head , head_dim)
        qkv = qkv.permute(2 , 0 , 3 , 1 , 4)  #(3 , n_samples , n_head , n_patches +1 , head_dim)
        q , k , v = qkv[0] , qkv[1] , qkv[2]  #(n_samples , n_head , n_patches + 1, head_dim)
        k_t = k.transpose(2 , 3) #(n_samples , n_head , head_dim , n_patches + 1)
        dp = (q @ k_t ) * self.scale #(n_samples , n_head , n_patches + 1 , n_patches + 1)
        attn_matrix = dp.softmax(dim = -1) #(n_samples , n_head , n_patches + 1 , n_patches + 1)
        attn = self.attn_drop(attn_matrix) #(n_samples , n_head  , n_patches + 1, n_patches + 1)
        weighted_avg = attn @ v #(N_samples , n_head , n_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(1 , 2) #(n_samples , n_patches +1 m n_head , head_dim)
        weighted_avg = weighted_avg.flatten(2) #(n_samples , n_patches + 1, embed_dim)
        x = self.proj(weighted_avg) #(n_samples , n_patches + 1 , embed_dim )
        output = self.proj_drop(x)
        return output





