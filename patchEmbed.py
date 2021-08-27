import torch 
import torch.nn as nn




class PatchEmbed(nn.Module):
    """
    Splits the image into patches and embed the patches 
    
    Parameters 
    ----------
    img_size : int 
        Size of input image(W == H)
    
    patch_size : int 
        Size of the patch(W == H)
    
    embed_dimension : int 
        Size of patch embedding   
    


    Attributes
    -----------

    n_patches : int 
        Number of patches of the image
    
    proj : nn.Conv2d
        Convolution to convert does both splitting into patches and embedding the patches
    """


    def __init__(self , img_size , patch_size ,in_channels = 3 , embed_dimensions = 768 ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels , embed_dimensions , kernel_size=patch_size , stride=patch_size)

    def forward(self , x):
        """
        Forward pass of the Module 

        Parameters
        ----------
        x : torch.Tensor
            Shape - (n_samples , input_channels , img_size , img_size)
        
        Output 
        -------
        torch.Tensor
            Shape - (n_samples , n_patches , embed_dimensions)        
        
        """
        x = self.proj(x) # (n_samples , embed_dim , n_patch ** 0.5 , n_patch ** 0.5)
        x = x.flatten(2) # (n_samples , embed_dim , n_patch)
        x = x.transpose(1 ,2 ) #(n_samples , n_patches , embed_dim)
        return x

    