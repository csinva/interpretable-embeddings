from torch import nn


class MLP(nn.Module):
    def __init__(self, dim_inputs=768, dim_hidden=100, dim_outputs=2, nonlin=nn.ReLU()):
        super(MLP, self).__init__()

        self.dense0 = nn.Linear(dim_inputs, dim_hidden)
        self.nonlin = nonlin
        self.dense1 = nn.Linear(dim_hidden, dim_outputs)
        # self.output = nn.Linear(n_outputs, n_outputs)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        # X = self.dropout(X)
        X = self.dense1(X)
        return X
