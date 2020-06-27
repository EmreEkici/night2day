import torch

from model import create_model, create_parallel_model, load_model, save_model, move_model_to_GPU
from dataset import create_data_loader
from utils import save_training_losses, get_training_losses, calculate_avg_epoch_losses

##############
model_num = 0
gpu_ids = (0, 1)
##############

# Create the model and use GPU if available
model = create_model()
#model = load_model("model_"+str(model_num)+"_epoch_2")
model, device = move_model_to_GPU(model)
#model = create_parallel_model(gpu_ids)

# Create the data generators
training_loader = create_data_loader("train")
validation_loader = create_data_loader("val")

epoch_count = 200
save_model_every = 5
print_losses_every = 100

losses_G = []
losses_D = []

print("Training the model...")

for epoch in range(1, epoch_count + 1):
    print("Epoch:", epoch)

    # Lists to calculate average epoch losses
    epoch_losses_G = []
    epoch_losses_D = []
    iter_count = 0

    for iter, (batch, labels) in enumerate(training_loader):
        batch, labels = batch.to(device), labels.to(device)
        
        model.set_input(batch, labels)
        model.optimize_parameters()
        model.update_learning_rates()

        # Keep track of iteration losses
        loss_G, loss_D = model.get_current_losses()

        epoch_losses_G.append(loss_G)
        epoch_losses_D.append(loss_D)
        iter_count += 1

        # Display current losses
        if (iter+1) % print_losses_every == 0:
            print("Iteration:", iter+1, "G_loss:", loss_G, "D_loss:", loss_D)

    # Save the model and the losses
    if (epoch+1) % save_model_every == 0:
        avg_epoch_loss_G, avg_epoch_loss_D = calculate_avg_epoch_losses(epoch_losses_G, epoch_losses_D, iter_count)

        losses_G.append(avg_epoch_loss_G)
        losses_D.append(avg_epoch_loss_D)

        save_training_losses(model_num, losses_G, losses_D)
        save_model(model, "model_"+str(model_num)+"_epoch_" + str(epoch))

print("Training done.")
