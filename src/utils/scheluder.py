import numpy as np


class GapAwareLRScheduler:
    def __init__(self, optimizer, ideal_loss, x_min, x_max, h_min=0.1, f_max=2.0):
        """
        Gap-aware Learning Rate Scheduler for Adversarial Networks.

        Args:
            optimizer: PyTorch optimizer.
            ideal_loss: the ideal loss of D.
            x_min: the value of x at which the scheduler achieves its minimum allowed value h_min.
            x_max: the value of x at which the scheduler achieves its maximum allowed value f_max.
            h_min: minimum allowed value of the scheduling function. Default is 0.1.
            f_max: maximum allowed value of the scheduling function. Default is 2.0.
        """
        self.optimizer = optimizer
        self.ideal_loss = ideal_loss
        self.smoothed_disc_loss = ideal_loss
        self.learning_rates = [group["lr"] for group in optimizer.param_groups]
        self.x_min = x_min
        self.x_max = x_max
        self.h_min = h_min
        self.f_max = f_max

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
        self.smoothed_disc_loss = 0.95 * self.smoothed_disc_loss + 0.05 * loss
        loss = self.smoothed_disc_loss

        x = np.abs(loss - self.ideal_loss)
        f_x = np.clip(np.power(self.f_max, x / self.x_max), 1.0, self.f_max)
        h_x = np.clip(np.power(self.h_min, x / self.x_min), self.h_min, 1.0)

        if loss > self.ideal_loss:
            s_x = f_x
        else:
            s_x = h_x

        for param_group, lr in zip(self.optimizer.param_groups, self.learning_rates):
            param_group["lr"] = lr * s_x

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
