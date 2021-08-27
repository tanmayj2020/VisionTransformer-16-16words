import torch 
import numpy as np 
from PIL import Image 
k = 10 

imagenet_labels = dict(enumerate(open("classes.txt")))



model = torch.load("model.pth")
model.eval()


img = (np.array(Image.open("cat.png")) / 128 ) - 1  #range in [-1 , 1]


inp = torch.from_numpy(img).permute(2 , 0 , 1).unsqueeze(0).to(torch.float32)
logits = model(inp)
probs = torch.nn.functional.softmax(logits , dim =-1) #shape (n_samples , probs)



top_probs , top_ics = probs[0].topk(k) # 0th sample(top K values with their index)

for i , (ix_ , prob_) in enumerate(zip(top_ics , top_probs)):
    ix = ix_.item()
    prob = prob_.item()
    cls = imagenet_labels[ix]
    print(f"{i} : {cls} --- {prob:.4f}")
