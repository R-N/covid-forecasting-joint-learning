from torch.optim.lr_scheduler import OneCycleLR as _OneCycleLR
from torch_lr_finder.lr_finder import ExponentialLR, LinearLR
from copy import deepcopy
from .util import calculate_prediction_interval, round_digits

class OneCycleLR:
    def __init__(self, optimizer, max_lr, steps_per_epoch, epochs, div_factor=25):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.div_factor = div_factor
        print("div_factor", self.div_factor)
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = epochs
        self.epochs = 0
        self.scheduler = None
        self.create()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def create(self):
        self.scheduler = _OneCycleLR(
            self.optimizer,
            max_lr=self.max_lr,
            div_factor=self.div_factor,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.max_epochs
        )
        return self.scheduler

    def step(self, step=1):
        ret = self.scheduler.step()
        self.epochs += 1
        if self.epochs >= self.max_epochs:
            self.create()
        return ret

class LRFinderResult:
    def __init__(
        self,
        best_loss=None,
        descend_lr_1=None,
        descend_lr_2=None,
        best_lr=None,
        last_lr=None
    ):
        self.best_loss = best_loss
        self.descend_lr_1 = descend_lr_1
        self.descend_lr_2 = descend_lr_2
        self.best_lr = best_lr
        self.last_lr = last_lr

    def round_digits(self, x, n_digits=0):
        return round_digits(x, n_digits=n_digits)

    @property
    def descend_lr(self):
        return self.descend_lr_1 if self.descend_lr_1 is not None else self.descend_lr_2


class LRFinder(object):
    """Learning rate range test.
    From https://github.com/davidtvs/pytorch-lr-finder
    """

    def __init__(
        self,
        objective,
        model,
        optimizer
    ):
        # Check if the optimizer is already attached to a scheduler
        self.optimizer = optimizer
        self._check_for_scheduler()

        self.objective = objective
        self.model = model


        # Save the original state of the model and optimizer so they can be restored if
        # needed
        self.model_init_state = deepcopy(self.model.state_dict())
        self.optimizer_init_state = deepcopy(self.optimizer.state_dict())
        self._clear_history()

    @property
    def result(self):
        return LRFinderResult(
            best_loss=self.best_loss,
            descend_lr_1=self.descend_lr_1,
            descend_lr_2=self.descend_lr_2,
            best_lr=self.best_lr,
            last_lr=self.last_lr
        )

    def reset_state(self):
        self.model.load_state_dict(deepcopy(self.model_init_state))
        self.optimizer.load_state_dict(deepcopy(self.optimizer_init_state))

    def _clear_history(self):
        self.lr_history = []
        self.loss_history = []
        self.best_loss = None
        self.descend_lr_1 = None
        self.descend_lr_2 = None
        self.best_lr = None
        self.last_lr = None
        self.best_epoch = 0

    def range_test(
        self,
        start_lr=None,
        end_lr=1000,
        num_iter=25,
        step_mode="exp",
        smooth_f=0.05,
        diverge_th=5,
        accumulation_steps=1,
        history_length=None,
        rise_patience=1
    ):
        self._clear_history()

        # Check if the optimizer is already attached to a scheduler
        self._check_for_scheduler()

        # Set the starting learning rate
        if start_lr:
            self._set_learning_rate(start_lr)

        # Initialize the proper learning rate policy
        if step_mode.lower() == "exp":
            lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter)
        elif step_mode.lower() == "linear":
            lr_schedule = LinearLR(self.optimizer, end_lr, num_iter)
        else:
            raise ValueError("expected one of (exp, linear), got {}".format(step_mode))

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError("smooth_f is outside the range [0, 1[")

        history_length = history_length or int(num_iter * 0.25)
        descended_1, descended_2 = False, False
        rise_counter = 0
        rise_patience = rise_patience or int(num_iter * 0.075)
        min_delta_0 = None
        raw_loss_history = []
        for iteration in range(num_iter):
            # Train on batch and retrieve loss
            loss = self.objective(scheduler=lr_schedule)

            # Update the learning rate
            lr = lr_schedule.get_last_lr()[0]
            self.lr_history.append(lr)
            # lr_schedule.step()

            # Track the best loss and smooth it if smooth_f is specified
            loss_0 = loss
            if iteration == 0:
                self.best_lr = lr
                self.best_loss = loss
                self.best_loss_1 = loss
                first_loss = loss
                self.loss_history.append(loss)
            else:
                if smooth_f > 0:
                    loss = smooth_f * loss + (1 - smooth_f) * self.loss_history[-1]
                self.loss_history.append(loss)

                mean, min_delta = calculate_prediction_interval(raw_loss_history[:history_length])
                if not (descended_1 and descended_2):
                    if (not descended_1) and loss - mean < -min_delta:
                        descended_1 = True
                        self.descend_lr_1 = lr
                    if (not descended_2) and loss - first_loss < -min_delta:
                        descended_2 = True
                        self.descend_lr_2 = lr

                if min_delta_0 is None:
                    min_delta_0 = min_delta
                    rise = False
                else:
                    rise = (loss - self.best_loss) > min_delta
                    if rise:
                        rise_counter += 1
                    else:
                        min_delta_0 = min_delta

            if loss < self.best_loss:
                self.best_lr = lr
                self.best_loss = loss
                self.best_epoch = iteration

            raw_loss_history.append(loss_0)
            self.loss_history.append(loss)

            if rise_counter >= rise_patience or loss > diverge_th * self.best_loss:
                break

            # Check if the loss has diverged; if it has, stop the test

            self.last_lr = lr

        print(f"Learning rate search finished. best_lr: {self.best_lr} at {self.best_epoch} epochs with loss={self.best_loss} after {iteration+1}/{num_iter} epochs")

    def _set_learning_rate(self, new_lrs):
        if not isinstance(new_lrs, list):
            new_lrs = [new_lrs] * len(self.optimizer.param_groups)
        if len(new_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                "Length of `new_lrs` is not equal to the number of parameter groups "
                + "in the given optimizer"
            )

        for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
            param_group["lr"] = new_lr

    def _check_for_scheduler(self):
        for param_group in self.optimizer.param_groups:
            if "initial_lr" in param_group:
                raise RuntimeError("Optimizer already has a scheduler attached to it")
