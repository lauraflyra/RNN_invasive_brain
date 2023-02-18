import torch
import torch.nn as nn

# https://stackoverflow.com/questions/61300023/how-to-combine-static-features-with-time-series-in-forecasting
# https://medium.com/omdena/time-series-classification-tutorial-combining-static-and-sequential-feature-modeling-using-ac18fe85c92c
# https://towardsdatascience.com/neural-networks-with-multiple-data-sources-ef91d7b4ad5a
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6427263/
# https://www.nature.com/articles/s41598-017-09766-1


class RNNOnly(nn.Module):
    """
    Create a recurrent neural network that receives the features calculated by py_neuromodulation as input,
    and outputs the grip force of the desired hand.
    """

    def __init__(self, input_size=1, output_size = 1, hidden_size=64, num_layers=1, batch_size = 1, LSTM=False, bidirectional=False, dropout=0.1):
        """
        input data should be format sequence_length, batch_size, input_size
        :param input_size: number of features that define each time-stamp of the input sequence
        :param output_size: number of features that define each time-stamp of the output sequence
        :param hidden_size: defines the size of the hidden state. If hidden_size=4,
        then hidden state at each time step is a vector of length 4
        :param num_layers: number of stacked RNNs
        :param LSTM: bool, if we use a LSTM or a regular RNN
        :param bidirectional: wheter the RNN layer is bidirectional or not.
        RNN has the limitation that it processes inputs in strict temporal order. This means current input has context
        of previous inputs but not the future. Bidirectional RNN ( BRNN ) duplicates the RNN processing chain so that
        inputs are processed in both forward and reverse time order. Since we're dealing with applications that aim at
        aDBS, in principle we stick to the temporal order.
        :param dropout: if there is a dropout layer after the RNN, with dropout probability = dropout
        """

        super(RNNOnly, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if not LSTM:
            self.rnn = nn.RNN(input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=bidirectional,
                                  dropout=dropout)
        else:
            self.rnn = nn.LSTM(input_size=self.input_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   bidirectional=bidirectional,
                                   dropout=dropout)

        # Compress output to the same dim as y (expected output)
        self.linear = nn.Linear(self.hidden_size, output_size)

        # hidden tensor has shape (D * num_layers, batch_size, hidden_size)
        self.D=1
        if bidirectional:
            self.D = 2
        self.hidden = torch.zeros(self.D*num_layers,batch_size, hidden_size)


    def forward(self, input):

        # batch_size = 1
        out, self.hidden = self.rnn(input, self.hidden)     # out.shape = (seq_length, batch_size, hidden_size)
        out = out.reshape(-1, self.hidden_size)
        # out.shape = (seq_length*batch_size, hidden_size) -> proper shape for input to self.linear
        out = self.linear(out)      # out.shape = (seq_length*batch_size, output_size)
        # output = out.unsqueeze(dim=1)   # Returns a new tensor with a dimension of size one inserted at the specified position.

        return out



