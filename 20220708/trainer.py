from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim


class Trainer():

    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    def _train(self, x, y, config):
        self.model.train()  # Train mode on

        # Shuffle before begin.
        # |x| = (train_data_size, input_size)
        # |y| = (train_data_size, input_size)

        indices = torch.randperm(x.size(0), device=x.device)
        x = torch.index_select(x, dim=0, index=indices).split(
            config.batch_size, dim=0)
        y = torch.index_select(y, dim=0, index=indices).split(
            config.batch_size, dim=0)

        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            # |y_hat_i| = (batch_size, output_size)
            loss_i = self.crit(y_hat_i, y_i.squeeze())
            # |y_i| = (batch_size, )

            # Initialize the gradients of the model.
            self.optimizer.zero_grad()
            loss_i.backward()

            self.optimizer.step()

            if config.verbose >= 2:
                print(
                    f"Train Iteration {i + 1} / {len(x)}: loss={float(loss_i):.4f}")

            total_loss += float(loss_i)

        return total_loss / len(x)

    def _validate(self, x, y, config):
        # Turn evaluation model on.
        # |x| = (valid_data_size, input_size)
        # |y| = (valid_data_size, )
        self.model.eval()

        # Turn on the no_grad mode to make more efficiently.
        with torch.no_grad():
            x = x.split(config.batch_size, dim=0)
            y = y.split(config.batch_size, dim=0)
            # |x| = (batch_size, input_size)
            # |y| = (batch_size, )

            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())
                # |y_hat_i| = (batch_size, output_size)
                # |y_i| = (batch_size, )

                if config.verbose >= 2:
                    print(
                        f"Valid Iteration {i + 1} / {len(x)}: loss={float(loss_i):.4f}")

                total_loss += float(loss_i)

            return total_loss / len(x)

    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        bast_model = None

        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)

            # You must use deep copy to take a snapshot of current best weights.

            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print(
                f"Epoch ({epoch_index + 1} / {config.n_epochs}): train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} lowest_loss={lowest_loss}")

        # Restore to best model.
        self.model.load_state_dict(best_model)
