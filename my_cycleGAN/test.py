import torch

from model import load_model, move_model_to_GPU
from dataset import create_data_loader

##############
model_num = 0
##############

test_loader = create_data_loader("test")

model = load_model("model_" + str(model_num) + "_epoch_219", isTrain=True) # Optimize et
#model = model.netG
#model.eval()
#print(model.loss_history_G)

model, device = move_model_to_GPU(model)

print("Testing the model...")

for iter, batch in enumerate(test_loader):
	batch = batch.to(device)

	with torch.no_grad():
		model.set_input(batch)
		model.forward()
		model.save_images(iter, len(batch))

print("Testing done.")