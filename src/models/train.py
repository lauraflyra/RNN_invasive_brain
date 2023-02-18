import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from network_RNN_only import RNNOnly
import pandas as pd
import numpy as np

PATH_FEATURES = "/home/lauraflyra/Documents/BCCN/Lab_Rotation_USC/Code/Data/py_neuromodulation_derivatives/sub-000_ses-right_task-force_run-3/sub-000_ses-right_task-force_run-3_FEATURES.csv"
data_features = pd.read_csv(PATH_FEATURES, index_col=0)
features = data_features.filter(like='ECOG').filter(like='bandpass_activity_low beta').to_numpy()
# feature size = time steps x feature numbers (for bandpass activity low beta)
time_steps, n_features = features.shape

behavior = data_features["MOV_LEFT_CLEAN"].to_numpy().reshape(-1, 1)
# behavior.shape = time_steps, 1

# There are two ways of doing this:
# 1st. giving part of the time series for training, and the rest for testing. Or doing cross validation, but regardless, giving the whole time series
# 2nd. epoching the data around the movements, creating batches, give some of the batches for training and the others for validation. -> maybe it would be somehow biased, since movement onset would always be at the same time
# Epoching the data around the movements actually could be better to avoid padding for different sequence lengths

train_idx_features = np.arange(np.round(0.5 * time_steps), dtype=int)
test_idx_features = np.arange(1 + train_idx_features[-1], time_steps)

feat_train = torch.from_numpy(features[train_idx_features]).reshape(-1, 1, n_features).float()
feat_test = torch.from_numpy(features[test_idx_features]).reshape(-1, 1, n_features).float()
mov_features_train = torch.from_numpy(behavior[train_idx_features]).float()
mov_features_test = torch.from_numpy(behavior[test_idx_features]).float()

model = RNNOnly(input_size=n_features, output_size=1, hidden_size=64, num_layers=2, batch_size=1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

n_epochs = 100
min_valid_loss = np.inf

train_loss = []
valid_loss = []

for e in range(n_epochs):
    model.train()

    model.hidden = torch.zeros(model.D * model.num_layers, 1, model.hidden_size)

    mov_features_train_pred = model(feat_train)
    loss = loss_fn(mov_features_train_pred, mov_features_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    train_loss.append(loss.item())

    model.eval()
    mov_features_test_pred = model(feat_test)
    val_loss = loss_fn(mov_features_test_pred, mov_features_test)
    valid_loss.append(val_loss.item())

    print(f'Epoch {e + 1},\n Training Loss: {train_loss[-1] }, Validation Loss: {valid_loss[-1]}')
    if min_valid_loss > valid_loss[-1]:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss[-1]:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss[-1]
        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pt')

        plt.plot(mov_features_test_pred.detach().numpy())
        plt.plot(mov_features_test.detach().numpy())
        plt.title(f'Epoch {e + 1},\n Training Loss: {train_loss[-1] }, Validation Loss: {valid_loss[-1]}')
        plt.show()


