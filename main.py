import torch.optim as optim
import torch
import Prediction as Prediction
from model import *
from Features import Imageloader as I
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Select GPU
print("Device:",'cuda' if torch.cuda.is_available() else 'cpu')
num_feature = 64 *64 *64
model = Model.OCRModel(num_feature=num_feature)
model = model.to(device)  # Move model to the GPU (if available)
optimizer = optim.Adam(model.parameters(), lr=0.001)
ctc_loss =Model.ctc_loss

#train Model
Train.train(model=model,train_loader=I.train_loader,optimizer=optimizer,ctc_loss=ctc_loss,device=device,num_epochs=30)

#Validate Model
Validate.validate(model=model,val_loader=I.val_loader,criterion=ctc_loss)

#Test Model
Test.test(model=model,device=device,test_loader=I.test_loader,criterion=ctc_loss)

#Save Model
torch.save(model.state_dict(),"OCR.pth")