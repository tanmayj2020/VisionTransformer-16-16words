import numpy as np 
import timm
import torch 
from vit import VIT


def n_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2):
    a1 , a2 = t1.detach().numpy() , t2.detach().numpy() 
    np.testing.assert_allclose(a1 , a2)


model_name = "vit_base_patch16_384"

model_official = timm.create_model(model_name , pretrained=True)
model_official.eval()
print(type(model_official))

model_custom = VIT(384 , 16 , 3 ,1000 , 768 , 12 , 12 , 4 , True)


for (n_o , p_o) , (n_c , p_c) in zip(model_official.named_parameters() , model_custom.named_parameters()):
    assert p_o.numel() == p_c.numel()
    print(f"{n_o} | {n_c}")
    p_c.data[:] = p_o.data
    assert_tensors_equal(p_c.data , p_o.data)

inp = torch.rand(1 , 3,  384 , 384)
res_c = model_custom(inp)
res_o = model_official(inp)

assert n_parameters(model_official) == n_parameters(model_custom)


assert_tensors_equal(res_c , res_o)


torch.save(model_custom , "model.pth")



