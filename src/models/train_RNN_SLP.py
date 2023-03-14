import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torchmetrics import R2Score
from network_RNN_SLP import RNNSLP
import pandas as pd
import numpy as np

from src.data.input_output import n_batches, time_steps, n_features, neural, behavior, updrs


dt = 100
time = np.array([1000 + i * dt for i in range(time_steps)])

model = RNNSLP(input_size_dynamic=n_features, input_size_static=1, output_size=1, batch_size=n_batches, LSTM=False,
               bidirectional=False, dropout=0.1)

train_idx_neural = np.arange(np.round(0.8 * time_steps), dtype=int)
test_idx_neural = np.arange(1 + train_idx_neural[-1], time_steps)

feat_train = neural[train_idx_neural]
feat_test = neural[test_idx_neural]
mov_neural_train = behavior[train_idx_neural]
mov_neural_test = behavior[test_idx_neural]

time_train = time[train_idx_neural]
time_test = time[test_idx_neural]

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
        model.hidden_1 = (torch.zeros(model.D * model.num_layers, n_batches, model.hidden_size_1),
                          torch.zeros(model.D * model.num_layers, n_batches, model.hidden_size_1))

        model.hidden_2 = (torch.zeros(model.D * model.num_layers, n_batches, model.hidden_size_2),
                          torch.zeros(model.D * model.num_layers, n_batches, model.hidden_size_2))
    else:
        model.hidden_1 = torch.zeros(model.D * model.num_layers, n_batches, model.hidden_size_1)
        model.hidden_2 = torch.zeros(model.D * model.num_layers, n_batches, model.hidden_size_2)

    mov_neural_train_pred = model(feat_train, updrs)
    loss = loss_fn(mov_neural_train_pred[:-1,:,:], mov_neural_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    train_loss.append(loss.item())

    model.eval()
    mov_neural_test_pred = model(feat_test, updrs)
    val_loss = loss_fn(mov_neural_test_pred[:-1,:,:], mov_neural_test)
    valid_loss.append(val_loss.item())

    print(f'Epoch {e + 1},\n Training Loss: {train_loss[-1]}, Validation Loss: {valid_loss[-1]}')
    if min_valid_loss > valid_loss[-1]:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss[-1]:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss[-1]
        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pt')
        r2score = R2Score()
        for batch in range(n_batches):
            r2 = r2score(mov_neural_test_pred[:-1,batch,:].reshape(-1,1), mov_neural_test[:,batch,:].reshape(-1,1))
            plt.plot(time_test / 1000, mov_neural_test_pred[:-1,batch,:].reshape(-1,1).detach().numpy(), label='predicted')
            plt.plot(time_test / 1000, mov_neural_test[:,batch,:].reshape(-1,1).detach().numpy(), label='test data', alpha=0.5)
            plt.title('Epoch {}, batch {}, Training Loss: {:.4f}, \nValidation Loss: {:.4f}, Validation r^2: {:.4f}'.format
                      (e + 1, batch, train_loss[-1], valid_loss[-1], r2), fontsize=15)
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
