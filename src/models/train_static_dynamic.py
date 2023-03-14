import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torchmetrics import R2Score
from network_RNN_SLP import RNNSLP
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


PATH_UPDRS = "/home/lauraflyra/Documents/BCCN/Decoding_Parkinsons_UPDRS/RNN_invasive_brain/src/data/df_updrs.csv"
df_updrs = pd.read_csv(PATH_UPDRS, index_col=0)
df_all_channels = np.load(
    "/home/lauraflyra/Documents/BCCN/Decoding_Parkinsons_UPDRS/RNN_invasive_brain/src/data/channel_all.npy",
    allow_pickle=True)
PATH_RMAP = "/home/lauraflyra/Documents/BCCN/Decoding_Parkinsons_UPDRS/RNN_invasive_brain/src/data/df_best_func_rmap_ch.csv"
df_rmap = pd.read_csv(PATH_RMAP, index_col=0)
data_s0_ECOG = torch.Tensor()
labels_s0_ECOG = torch.Tensor()
for sess in list(df_all_channels[()]['Pittsburgh']['000']['ECOG_RIGHT_0-avgref'].keys()):
    data_s0_ECOG = torch.cat(
        (data_s0_ECOG, torch.Tensor(df_all_channels[()]['Pittsburgh']['000']['ECOG_RIGHT_0-avgref'][sess]['data'])),
        axis=0)
    labels_s0_ECOG = torch.cat(
        (labels_s0_ECOG, torch.Tensor(df_all_channels[()]['Pittsburgh']['000']['ECOG_RIGHT_0-avgref'][sess]['label'])),
        axis=0)

time_steps, n_features = data_s0_ECOG.shape
features = data_s0_ECOG.reshape(time_steps, 1, n_features)
behavior = (labels_s0_ECOG >= 0.5).long().reshape(-1, 1,1).float()
behavior = labels_s0_ECOG.reshape(-1, 1,1).float()
dt = 100
time = np.array([1000 + i * dt for i in range(time_steps)])

updrs = torch.tensor(df_updrs[(df_updrs['cohort'] == 'Pittsburgh') & (df_updrs['sub'] == '000')]['UPDRS_total']).float()

model = RNNSLP(input_size_dynamic=n_features, input_size_static=1, output_size=1, batch_size=1, LSTM=False,
               bidirectional=False, dropout=0.1)

train_idx_features = np.arange(np.round(0.5 * time_steps), dtype=int)
test_idx_features = np.arange(1 + train_idx_features[-1], time_steps)

feat_train = features[train_idx_features]
feat_test = features[test_idx_features]
mov_features_train = behavior[train_idx_features]
mov_features_test = behavior[test_idx_features]

time_train = time[train_idx_features]
time_test = time[test_idx_features]

loss_fn = nn.MSELoss()

optimizer_grouped_parameters = [
    {'params': model.params.rnn2.parameters(), 'weight_decay':0.01},
    {'params': model.params.static.parameters(), 'weight_decay': 0.001},
    {'params': model.params.base.parameters()},
]

optimizer = torch.optim.Adam(optimizer_grouped_parameters)

n_epochs = 100
min_valid_loss = np.inf

train_loss = []
valid_loss = []

for e in range(n_epochs):
    model.train()

    if model.LSTM:
        model.hidden_1 = (torch.zeros(model.D * model.num_layers, 1, model.hidden_size_1),
                          torch.zeros(model.D * model.num_layers, 1, model.hidden_size_1))

        model.hidden_2 = (torch.zeros(model.D * model.num_layers, 1, model.hidden_size_2),
                          torch.zeros(model.D * model.num_layers, 1, model.hidden_size_2))
    else:
        model.hidden_1 = torch.zeros(model.D * model.num_layers, 1, model.hidden_size_1)
        model.hidden_2 = torch.zeros(model.D * model.num_layers, 1, model.hidden_size_2)

    mov_features_train_pred = model(feat_train, updrs)
    loss = loss_fn(mov_features_train_pred[:-1,:,:], mov_features_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    train_loss.append(loss.item())

    model.eval()
    mov_features_test_pred = model(feat_test, updrs)
    val_loss = loss_fn(mov_features_test_pred[:-1,:,:], mov_features_test)
    valid_loss.append(val_loss.item())

    print(f'Epoch {e + 1},\n Training Loss: {train_loss[-1]}, Validation Loss: {valid_loss[-1]}')
    if min_valid_loss > valid_loss[-1]:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss[-1]:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss[-1]
        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pt')
        r2score = R2Score()
        r2 = r2score(mov_features_test_pred[:-1,:,:].reshape(-1,1), mov_features_test.reshape(-1,1))
        plt.plot(time_test / 1000, mov_features_test_pred[:-1,:,:].reshape(-1,1).detach().numpy(), label='predicted')
        plt.plot(time_test / 1000, mov_features_test.reshape(-1,1).detach().numpy(), label='test data', alpha=0.5)
        plt.title('Epoch {}, Training Loss: {:.4f}, \nValidation Loss: {:.4f}, Validation r^2: {:.4f}'.format
                  (e + 1, train_loss[-1], valid_loss[-1], r2), fontsize=15)
        plt.xlabel('Time [s]', fontsize=15)
        plt.ylabel('Grip force', fontsize=15)
        plt.legend()
        plt.show()

plt.plot(np.arange(0, n_epochs), train_loss, label='training')
plt.plot(np.arange(0, n_epochs), valid_loss, label='validation')
plt.ylabel('MSE', fontsize=15)
plt.xlabel('Epochs', fontsize=15)
plt.legend()
plt.title("Error over epochs in training with \n RNN + ECoG bandpower low beta", fontsize=15)
plt.show()
