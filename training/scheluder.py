import numpy as np


class GapAwareLRScheduler:
    def __init__(self, optimizer, ideal_loss, x_min, x_max, h_min=0.1, f_max=2.0):
        self.optimizer = optimizer
        self.ideal_loss = ideal_loss
        self.smoothed_disc_loss = ideal_loss
        self.learning_rates = [group["lr"] for group in optimizer.param_groups]
        self.x_min = x_min
        self.x_max = x_max
        self.h_min = h_min
        self.f_max = f_max

        # for param_group, lr in zip(self.optimizer.param_groups, self.learning_rates):
        #     param_group["lr"] = 0.0002

    def zero_grad(self):
        """
        Zero the gradients of the optimizer.
        """
        self.optimizer.zero_grad()

    def step(self, loss):
        """
        Update the learning rate of the optimizer based on the current loss.

        Args:
            loss: current loss of the discriminator.
        """
        self.smoothed_disc_loss = 0.999 * self.smoothed_disc_loss + 0.001 * loss
        loss = self.smoothed_disc_loss

        x = np.abs(loss - self.ideal_loss)
        f_x = np.clip(np.power(self.f_max, x / self.x_max), 1.0, self.f_max)
        h_x = np.clip(np.power(self.h_min, x / self.x_min), self.h_min, 1.0)

        if loss > self.ideal_loss:
            s_x = f_x
        else:
            s_x = h_x

        # for param_group, lr in zip(self.optimizer.param_groups, self.learning_rates):
        #     param_group["lr"] = lr * s_x

        return s_x

    def state_dict(self):
        """
        Return the state of the scheduler as a dictionary.
        """
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        """
        Load the state of the scheduler from a dictionary.

        Args:
            state_dict: dictionary containing the state of the scheduler.
        """
        self.optimizer.load_state_dict(state_dict["optimizer"])


class LinearWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, min_lr=0.0):
        """
        Linear Warmup Scheduler for Adversarial Networks.

        Args:
            optimizer: PyTorch optimizer.
            warmup_steps: number of steps for the warmup phase.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps if warmup_steps > 0 else 1
        self.min_lr = min_lr
        self.learning_rates = [group["lr"] for group in optimizer.param_groups]
        self.steps = 1

        for param_group, lr in zip(self.optimizer.param_groups, self.learning_rates):
            param_group["lr"] = min_lr if lr < min_lr and warmup_steps > 0 else lr

    def zero_grad(self):
        """
        Zero the gradients of the optimizer.
        """
        self.optimizer.zero_grad()

    def step(self):
        """
        Update the learning rate of the optimizer based on the current step.

        Args:
            step: current step of the training.
        """
        self.steps += 1
        if self.steps <= self.warmup_steps:
            new_lrs = [
                self.min_lr + (lr - self.min_lr) * (self.steps / self.warmup_steps)
                for lr in self.learning_rates
            ]

            for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
                param_group["lr"] = new_lr

    def state_dict(self):
        """
        Return the state of the scheduler as a dictionary.
        """
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        """
        Load the state of the scheduler from a dictionary.

        Args:
            state_dict: dictionary containing the state of the scheduler.
        """
        self.optimizer.load_state_dict(state_dict["optimizer"])
