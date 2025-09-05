import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchvision.models as models 
import torchvision.transforms as transforms 
from PIL import Image 
import copy 
import matplotlib.pyplot as plt 
import torch.nn.functional as F 

cnn = models.vgg19(pretrained = True).features

#DATA PREPROCESSING 
def image_loader(image_path):
    image = Image.open(image_path).convert("RGB")
    loader = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3,:,:]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])

                                ])
    return loader(image).unsqueeze(0)
#loading the image

content_img_path = "images/content/boston.jpg"
style_img_path = "images/style/vangogh_starry_night.jpg"

#PREPROCESSING THE IMAGE

content_img = image_loader(content_img_path)
style_img = image_loader(style_img_path)

print(content_img.shape)
print(style_img.shape)




#CONTENT LOSS  

def gram_matrix(x): 
    b , c , h , w  = x.size() #batch , channels , height , width
    features = x.view(b*c , h*w) 
    G = torch.mm( features , features.t())
    return G.div(b*c*h*w) ##try returning only G 
    #return G

class ContentLoss(nn.Module):
    def __init__(self , target): 
        super(ContentLoss,self).__init__()
        self.target = target.detach()
        self.loss = 0
    def forward(self , x): 
        self.loss = F.mse_loss(x , self.target)
        return x


class StyleLoss(nn.Module):
    def __init__(self , target_features): 
        super(StyleLoss , self).__init__()
        self.target = gram_matrix(target_features).detach() 
        self.loss = 0 

    def forward(self , x ): 
        G = gram_matrix(x)
        self.loss = F.mse_loss(G , self.target)
        return x 


cnn_copy = copy.deepcopy(cnn)

content_losses = [] 
style_losses = [] 


model = nn.Sequential()

content_layers = ['conv_4']  # typical choice
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


##we are going throught VGG19 layer by layer 
i = 0 
for layer in cnn_copy.children(): 
    

    if isinstance(layer , nn.Conv2d):
        i += 1 
        name = f'conv_{i}'
    elif isinstance(layer , nn.ReLU):
        name = f'Relu_{i}'
        layer = nn.ReLU(inplace = False)
    elif isinstance(layer , nn.MaxPool2d):
        name = f'pool_{i}'
    elif isinstance(layer , nn.BatchNorm2d):
        name = f'bn_{i}'
    else : 
        print("UNrecognizsed layer")
        
    model.add_module(name,layer)

    if name in content_layers:
        target = model(content_img).detach()
        content_loss = ContentLoss(target)
        model.add_module(f"content_loss_{i}", content_loss)
        content_losses.append(content_loss)

    if name in style_layers:
        target_feature = model(style_img).detach()
        style_loss = StyleLoss(target_feature)
        model.add_module(f"style_loss_{i}", style_loss)
        style_losses.append(style_loss)





gen_img = content_img.clone() 
gen_img.requires_grad_(True)


optimizer = optim.LBFGS([gen_img]) #use admas optimizer once

num_steps = 300 
style_weight = 1e6 
content_weight = 1 

for step in range(num_steps):
    def closure(): 
        optimizer.zero_grad()
        model(gen_img)
        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        loss = style_weight* style_score + content_weight * content_score
        loss.backward()
        return loss 
    optimizer.step(closure) 




#Denormalize and Convert to image 
#So That we can seee it 


def  im_convert(tensor): 

    image = tensor.clone().detach() 
    image = image.squeeze(0)
    image = image.cpu() 


    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)

    image = image.clamp(0,1)  # make sure all values are between 0 and 1
    image = image.permute(1, 2, 0).numpy()  # convert to HWC for matplotlib
    return image


plt.figure(figsize=(8,8))
plt.imshow(im_convert(gen_img))
plt.axis('off')
plt.show()







