import torch
import torch.nn as nn


class Block(nn.Module):

    def __init__(self, input_size, output_size, use_batch_norm=True, dropout_p=.4):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p

        super().__init__()

        def get_regularizer(use_batch_norm, size):
            # before layer로부터 들어오는 feature size = size
            # flatten된 data를 사용하기 때문에 batchnorm1d를 사용한다.
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)

        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            # input_size - before layer의 output_size
            nn.LeakyReLU(),
            get_regularizer(use_batch_norm, output_size),
        )

    def forward(self, x):
        # |x| = (batch_size, input_size)
        # |y| = (batch_size, output_size)
        y = self.block(x)
        return y


class ImageClassifier(nn.Module):

    def __init__(self, input_size, output_size, hidden_sizes=[500, 400, 300, 200, 100], use_batch_norm=True, dropout_p=.3):

        super().__init__()

        assert len(hidden_sizes) > 0, 'You need to specify hidden layers'

        last_hidden_size = input_size
        blocks = []
        for hidden_size in hidden_sizes:
            blocks += [Block(
                last_hidden_size,
                hidden_size,
                use_batch_norm,
                dropout_p
            )]
            last_hidden_size = hidden_size

        self.layers = nn.Sequential(
            *blocks,
            nn.Linear(last_hidden_size, output_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.layers(x)
        # |y| = (batch_size, output_size)

        return y
