import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torchmetrics import R2Score
from network_RNN_only import RNNOnly
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold

"""
Define train function that can be used with all models implemented here. 
Optimizer is Adam. 
"""

def train(features,
          behavior,
          model_class,
          k_fold_cv = 2,
          loss_fn = nn.MSELoss(),
          n_epochs = 100,
          save_model = True,
          save_model_name = "model"):
    """

    :param features:
    :param behavior:
    :param model_class:
    :param k_fold_cv: int, k_fold cross validation.
    :param loss_fn:
    :param n_epochs:
    :param save_model:
    :param save_model_name:
    :return:
    """

    kfold = KFold(n_splits = k_fold_cv, shuffle = False)

    train_loss_array = np.zeros((k_fold_cv, n_epochs))
    valid_loss_array = np.zeros((k_fold_cv, n_epochs))


    for fold, (train_idx, valid_idx) in enumerate(kfold.split(features)):
        train_x_set = features[train_idx]
        valid_x_set = features[valid_idx]

        train_y_set = behavior[train_idx]
        valid_y_set = behavior[valid_idx]

        model = model_class
        optimizer = torch.optim.Adam(model.parameters())
        min_valid_loss = np.inf

        for e in range(n_epochs):

            model.train()

            if model.__class__.__name__ == 'RNNOnly':
                if model.LSTM:
                    model.hidden = (torch.zeros(model.D * model.num_layers, 1, model.hidden_size),
                                    torch.zeros(model.D * model.num_layers, 1, model.hidden_size))
                else:
                    model.hidden = torch.zeros(model.D * model.num_layers, 1, model.hidden_size)

            # training
            train_y_pred = model(train_x_set)
            train_loss = loss_fn(train_y_pred, train_y_set)
            optimizer.zero_grad()
            train_loss.backward()
            train_loss_array[fold, e] = train_loss.item()
            # validation
            model.eval()
            valid_y_pred = model(valid_x_set)
            valid_loss = loss_fn(valid_y_pred, valid_y_set)
            valid_loss_array[fold, e] = valid_loss.item()

            if save_model:
                if min_valid_loss > valid_loss.item():
                    print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss.item():.6f}) \t Saving The Model')
                    min_valid_loss = valid_loss.item()
                    # Saving State Dict
                    save_model_name = save_model_name + str(fold) + ".pt"
                    torch.save(model.state_dict(), save_model_name)
                    # r2score = R2Score()
                    # r2 = r2score(valid_y_pred, valid_y_set)
                    plt.plot(valid_y_pred.detach().numpy(), label='predicted')
                    plt.plot(valid_y_set.detach().numpy(), label='test data')
                    plt.title(
                        'Fold {}, Epoch {}, Training Loss: {:.4f}, \nValidation Loss: {:.4f}'.format
                        (fold, e + 1, train_loss.item(), valid_loss.item()), fontsize=15)
                    plt.xlabel('Time [s]', fontsize=15)
                    plt.ylabel('Grip force', fontsize=15)
                    plt.legend()
                    plt.show()


    plt.plot(np.arange(0, n_epochs), np.mean(train_loss_array,axis=0), label='training')
    plt.plot(np.arange(0, n_epochs), np.mean(valid_loss_array,axis=0), label='validation')
    plt.ylabel('MSE', fontsize=15)
    plt.xlabel('Epochs', fontsize=15)
    plt.legend()
    plt.title("Error over epochs in training with \n RNN + ECoG bandpower low beta", fontsize=15)
    plt.show()
    return train_loss_array, valid_loss_array


PATH_UPDRS = "/home/lauraflyra/Documents/BCCN/Decoding_Parkinsons_UPDRS/RNN_invasive_brain/src/data/df_updrs.csv"
df_updrs = pd.read_csv(PATH_UPDRS, index_col=0)
df_all_channels = np.load("/home/lauraflyra/Documents/BCCN/Decoding_Parkinsons_UPDRS/RNN_invasive_brain/src/data/channel_all.npy", allow_pickle=True)
PATH_RMAP = "/home/lauraflyra/Documents/BCCN/Decoding_Parkinsons_UPDRS/RNN_invasive_brain/src/data/df_best_func_rmap_ch.csv"
df_rmap = pd.read_csv(PATH_RMAP, index_col=0)
data_s0_ECOG = torch.Tensor()
labels_s0_ECOG = torch.Tensor()
for sess in list(df_all_channels[()]['Pittsburgh']['000']['ECOG_RIGHT_0-avgref'].keys()):
    data_s0_ECOG = torch.cat((data_s0_ECOG, torch.Tensor(df_all_channels[()]['Pittsburgh']['000']['ECOG_RIGHT_0-avgref'][sess]['data'])), axis=0)
    labels_s0_ECOG = torch.cat((labels_s0_ECOG, torch.Tensor(df_all_channels[()]['Pittsburgh']['000']['ECOG_RIGHT_0-avgref'][sess]['label'])), axis = 0)

time_points, n_features = data_s0_ECOG.shape
data_s0_ECOG = data_s0_ECOG.reshape(time_points, 1, n_features)
labels_s0_ECOG = labels_s0_ECOG.reshape(-1, 1)


# TODO: there's something wrong, that the model doesnt seem to learn between epochs
# try train_v2 with this data

train(features=data_s0_ECOG,
      behavior=labels_s0_ECOG,
      model_class=RNNOnly(input_size=n_features, output_size=1, hidden_size=64, num_layers=2, batch_size=1, LSTM=False),
      k_fold_cv=5
      )


