import torch
import torch.nn as nn


# https://stackoverflow.com/questions/61300023/how-to-combine-static-features-with-time-series-in-forecasting
# https://medium.com/omdena/time-series-classification-tutorial-combining-static-and-sequential-feature-modeling-using-ac18fe85c92c
# https://towardsdatascience.com/neural-networks-with-multiple-data-sources-ef91d7b4ad5a
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6427263/
# https://www.nature.com/articles/s41598-017-09766-1


class RNNSLP(nn.Module):
    """
    Create a recurrent neural network that receives the features calculated by py_neuromodulation as input,
    and outputs the grip force of the desired hand.
    RNN + single layer perceptron
    """

    def __init__(self, input_size_dynamic=1, input_size_static=1, output_size=1, batch_size=1, LSTM=False,
                 bidirectional=False, dropout=0.1):

        super(RNNSLP, self).__init__()
        # we're gonna have 2 RNN layers, the first one with 128 as hidden dim, and the second 64
        # both have dropout prob 0.1
        # can be either normal RNN or LSTM

        self.input_size_dynamic = input_size_dynamic
        self.input_size_static = input_size_static
        self.output_size = output_size
        self.LSTM = LSTM

        self.hidden_size_1 = 128
        self.hidden_size_2 = 64

        self.D = 1
        if bidirectional:
            self.D = 2

        self.num_layers = 2

        if not self.LSTM:
            self.rnn_1 = nn.RNN(input_size=self.input_size_dynamic,
                                hidden_size=self.hidden_size_1,
                                num_layers=self.num_layers,  # 2 layers because of dropout
                                bidirectional=bidirectional,
                                dropout=dropout)

            self.rnn_2 = nn.RNN(input_size=self.hidden_size_1,
                                hidden_size=self.hidden_size_2,
                                num_layers=self.num_layers,
                                bidirectional=bidirectional,
                                dropout=dropout)

            self.hidden_1 = torch.zeros(self.D*self.num_layers, batch_size, self.hidden_size_1)
            self.hidden_2 = torch.zeros(self.D*self.num_layers, batch_size, self.hidden_size_2)
        else:
            self.rnn_1 = nn.LSTM(input_size=self.input_size_dynamic,
                                 hidden_size=self.hidden_size_1,
                                 num_layers=2,
                                 bidirectional=bidirectional,
                                 dropout=dropout)

            self.rnn_2 = nn.LSTM(input_size=self.hidden_size_1,
                                 hidden_size=self.hidden_size_2,
                                 num_layers=self.num_layers,
                                 bidirectional=bidirectional,
                                 dropout=dropout)

            self.hidden_1 = (torch.zeros(self.D*self.num_layers, batch_size, self.hidden_size_1),
                             torch.zeros(self.D*self.num_layers, batch_size, self.hidden_size_1))

            self.hidden_2 = (torch.zeros(self.D*self.num_layers, batch_size, self.hidden_size_2),
                             torch.zeros(self.D*self.num_layers, batch_size, self.hidden_size_2))


        self.static = nn.Linear(self.input_size_static, self.hidden_size_2)
        self.relu = nn.ReLU()

        # Combine layers - RNN + SLP
        self.combined_layer = nn.Linear(self.hidden_size_2, self.hidden_size_2)

        self.output = nn.Linear(self.hidden_size_2, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()

        # create parameter groups for optimization
        self.params = nn.ModuleDict({
            'rnn2': nn.ModuleList([self.rnn_2]),
            'static': nn.ModuleList([self.static]),
            'base': nn.ModuleList([self.rnn_1, self.combined_layer, self.output])
        })


    def forward(self, input_dynamic, input_static):

        out_rnn_1, self.hidden_1 = self.rnn_1(input_dynamic, self.hidden_1)
        out_rnn_2, self.hidden_2 = self.rnn_2(out_rnn_1, self.hidden_2)

        out_static_temp = self.static(input_static)
        out_static = self.relu(out_static_temp)

        combined_input = torch.cat((out_rnn_2, out_static.reshape(1,-1,self.hidden_size_2)), axis=0)

        out_combined_temp = self.combined_layer(combined_input)
        out_combined = self.relu(out_combined_temp)

        out_final_temp = self.output(out_combined)
        # classification problem is not working!
        # out_final = self.sigmoid(out_final_temp)

        return out_final_temp
