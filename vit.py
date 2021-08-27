import torch 
import torch.nn as nn
from patchEmbed import PatchEmbed
from block import Block

class VIT(nn.Module):
    """
    Vision Transformer Class

    Parameters
    ----------
    img_size : int
        Both height and width of the image(square)

    patch_size : int 
        Both height and width of the patch (sqaure)
    in_channels : 
        Number of input channels   
    n_classes : int
        Number of classes
    mlp_ratio : float 
        Hidden layer ration of the MLP module 
    qkv_bias : bool     
        If true then we include bias to query , key and value matrix
    p , attn_p : float 
        Dropout Probability 

        
    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of PatchEmbed layer
    
    cls_token : nn.Parameter
        Learnable parameter that represents the first token in the sequence 
        "It has dimensions of embed dimensions"

    pos_emb : nn.Parameter
        Positional embedding of cls token + all the the patches
        It has (n_patches +1 ) * embed_dim elements

    pos_drop : nn.Dropout 
        Dropout Layer
    blocks : nn.ModuleList 
        List of 'Block' module
    norm : nn.LayerNorm 
        Layer Normalisation 
    """
    def __init__(self , img_size = 384 , patch_size = 16 , in_chans = 3 , n_classes = 1000 , embed_dim = 768 , depth = 12 , n_heads = 12 , mlp_ratio = 4. , qkv_bias = True , p = 0. , attn_p = 0.):
        super().__init__()
        self.patch_embed =  PatchEmbed(img_size , patch_size , in_chans , embed_dim )
        self.cls_token = nn.Parameter(torch.zeros(1 ,1 , embed_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1 , 1 + self.patch_embed.n_patches , embed_dim))
        self.pos_drop = nn.Dropout(p = p)
        self.blocks  = nn.ModuleList(
            [Block(embed_dim , n_heads , mlp_ratio , qkv_bias , p , attn_p) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim , eps =1e-6)
        self.head = nn.Linear(embed_dim , n_classes)
    

    def forward(self , x):
        """
        Run the forward pass

        Parameters 
        ----------
        x : torch.Tensor
            Shape = (n_samples , in_channels , img_size , img_size)

        Returns 
        -------
        logits : torch.Tensor 
            Logits over all classes Shape = (n_samples , n_classes)            
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x) #(n_samples , n_patches , embed_dim)
        cls_token = self.cls_token.expand(n_samples , - 1, -1) #(n_samples , 1 , embed_dim)
        x = torch.cat((cls_token , x) , dim = 1) #(n_samples , 1 + n_patches, embed_dim)
        x = x + self.pos_emb #(n_samples , n_patches + 1, embed_dim)
        x = self.pos_drop(x) #(n_samples , n_patches +1 , embed_dim)
        for blocks in self.blocks:
            x = blocks(x)    #(n_samples, n_patches + 1, embed_dim)

        x = self.norm(x) #(n_samples, n_patches + 1, embed_dim)
        cls_token_final = x[: ,0] #Just the class tokens (n_samples ,1 , embed_dim)
        x = self.head(cls_token_final) #(n_samples, n_class)
        return x



        
