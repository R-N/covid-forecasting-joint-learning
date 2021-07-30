from .util import progressive_smooth
import scipy.stats as st
from math import sqrt


class EarlyStopping:
    def __init__(
        self, 
        model,
        wait=20, wait_train_below_val=20, 
        rise_patience=10, still_patience=6, 
        min_delta_val=1.5e-5, min_delta_val_percent=0.15, 
        min_delta_train=2e-6, min_delta_train_percent=0.025, 
        smoothing=0.6,
        debug=0
    ):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.model = model
        self.wait = wait
        self.wait_counter = 0
        self.wait_train_below_val = wait_train_below_val
        self.wait_train_below_val_counter = 0
        self.train_below_val = False
        self.rise_patience = rise_patience
        self.still_patience = still_patience
        self.min_delta_val = min_delta_val
        self.min_delta_val_percent = min_delta_val_percent
        self.min_delta_train = min_delta_train
        self.min_delta_train_percent = min_delta_train_percent
        self.rise_counter = 0
        self.still_counter = 0
        self.last_train_loss = None
        self.last_val_loss = None
        self.best_val_loss = None
        self.best_train_loss = None
        self.best_val_loss_2 = None
        self.early_stopped = False
        self.smoothing = min(1.0, max(0, 1.0 - smoothing))
        self.debug = debug

    def __call__(self, train_loss, val_loss):
        # self.train_loss_history = [*self.train_loss_history, train_loss][-self.smoothing:]
        # self.val_loss_history = [*self.val_loss_history, val_loss][-self.smoothing:]
        if self.last_val_loss is not None:
            train_loss = progressive_smooth(self.last_train_loss, self.smoothing, train_loss)
            val_loss = progressive_smooth(self.last_val_loss, self.smoothing, val_loss)
        self.last_train_loss = train_loss
        self.last_val_loss = val_loss

        if self.wait_counter < self.wait:
            self.wait_counter += 1
            if self.debug >= 3:
                print(f"INFO: Early stopping wait {self.wait_counter}/{self.wait}")
        elif not self.train_below_val and val_loss < train_loss and self.wait_train_below_val_counter < self.wait_train_below_val:
            self.wait_train_below_val_counter += 1
            if self.debug >= 3:
                print(f"INFO: Early stopping wait train below val {self.wait_train_below_val_counter}/{self.wait_train_below_val}")
        else:
            self.train_below_val = True
            if self.best_val_loss is None:
                self.update_best_2(val_loss)
                self.update_best(train_loss, val_loss)
            else:
                delta_val_loss = val_loss - self.best_val_loss
                delta_train_loss = train_loss - self.best_train_loss
                if self.debug >= 3:
                    print("INFO: Early stopping check", train_loss, val_loss, delta_train_loss, delta_val_loss)
                min_delta_val_percent = self.min_delta_val_percent * val_loss
                min_delta_val = max(self.min_delta_val, min_delta_val_percent)
                rise = delta_val_loss > min_delta_val
                if rise:
                    self.rise_counter += 1
                    self.still_counter = 0  # It will need time to go down
                    if self.debug >= 2:
                        print(f"INFO: Early stopping rise {self.rise_counter}/{self.rise_patience}")
                    if self.rise_counter >= self.rise_patience:
                        self.early_stop("rise")
                else:
                    self.rise_counter = 0
                    still = abs(delta_val_loss) < min_delta_val
                    min_delta_train_percent = self.min_delta_train_percent * train_loss
                    min_delta_train = max(self.min_delta_train, min_delta_train_percent)
                    if still:
                        if val_loss < self.best_val_loss_2:
                            self.still_counter += 0.5
                            self.update_best_2(val_loss)
                        elif delta_train_loss < -min_delta_train:
                            self.still_counter += 0.75
                        else:
                            self.still_counter += 1
                        if self.debug >= 2:
                            print(f"INFO: Early stopping still {self.still_counter}/{self.still_patience}")
                        if self.still_counter >= self.still_patience:
                            self.early_stop("still")
                    else:
                        self.update_best(train_loss, val_loss)
                        self.still_counter = 0
        return self.early_stopped

    def update_state(self):
        self.best_state = self.model.state_dict()

    def update_best_2(self, val_loss):
        self.best_val_loss_2 = val_loss
        self.update_state()

    def update_best(self, train_loss, val_loss):
        self.best_train_loss = train_loss
        self.best_val_loss = val_loss
        if val_loss < self.best_val_loss_2:
            self.update_best_2(val_loss)

    def early_stop(self, reason="idk"):
        self.model.load_state_dict(self.best_state)
        self.early_stopped = True
        if self.debug >= 1:
            print(f"INFO: Early stopping due to {reason}")


class EarlyStopping2:
    def __init__(
        self, 
        model,
        wait=20, wait_train_below_val=20, 
        rise_patience=10, still_patience=6,
        interval_percent=0.05,
        min_delta_val=1.5e-5, min_delta_train=2e-6,
        min_min_delta_val=1e-3, min_min_delta_train=1e-4,
        history_length=None,
        smoothing=0.6,
        simple=False,
        debug=0,
        log_dir=None,
        label=None
    ):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.model = model
        self.wait = wait
        self.wait_counter = 0
        self.wait_train_below_val = wait_train_below_val
        self.wait_train_below_val_counter = 0
        self.train_below_val = False
        self.rise_patience = rise_patience
        self.still_patience = still_patience
        self.min_delta_val = min_delta_val
        self.min_delta_train = min_delta_train
        self.min_min_delta_val = min_min_delta_val
        self.min_min_delta_train = min_min_delta_train
        self.rise_counter = 0
        self.still_counter = 0
        self.history_length = history_length or min(rise_patience, still_patience)
        self.train_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = None
        self.best_train_loss = None
        self.best_val_loss_2 = None
        self.early_stopped = False
        self.smoothing = min(1.0, max(0, 1.0 - smoothing))
        self.interval_percent = interval_percent
        self.debug = debug
        self.calculate_interval = self.calculate_interval_1 if simple else self.calculate_interval_2
        self.log_dir = log_dir
        self.label = label
        self.epoch = 0

        if self.log_dir is not None or self.label is not None:
            assert self.log_dir is not None and self.label is not None

            self.min_percent_high_writer = SummaryWriter(log_dir + "/min_percent_high")
            self.min_percent_low_writer = SummaryWriter(log_dir + "/min_percent_low")

            self.min_high_writer = SummaryWriter(log_dir + "/min_high")
            self.min_low_writer = SummaryWriter(log_dir + "/min_low")

            self.loss_writer = SummaryWriter(log_dir + "/loss")
            self.best_loss_writer = SummaryWriter(log_dir + "/best_loss")
            self.best_loss_2_writer = SummaryWriter(log_dir + "/best_loss_2")

            self.still_writer = SummaryWriter(log_dir + "/still")
            self.rise_writer = SummaryWriter(log_dir + "/rise")

    def calculate_interval_1(self, history):
        min_val = min(history)
        max_val = max(history)
        delta = 0.5 * (1.0 - self.interval_percent) * (max_val - min_val)
        mid = 0.5 * (min_val + max_val)
        return mid, delta

    def calculate_interval_2(self, history):
        mean = sum(history)/self.history_length
        sum_err = sum([(mean-x)**2 for x in history])
        stdev = sqrt(1 / (self.history_length - 2) * sum_err)
        mul = st.norm.ppf(1.0 - self.interval_percent) if self.interval_percent >= 0 else 2 + self.interval_percent
        sigma = mul * stdev
        return mean, sigma

    def log_stop(
        self,
        label,epoch,
        loss, 
        min_delta, min_delta_percent,
        best_loss, best_loss_2=None
    ):
        self.min_percent_high_writer.add_scalar(self.label + label, best_loss + min_delta_percent, global_step=epoch)
        self.min_percent_low_writer.add_scalar(self.label + label, best_loss - min_delta_percent, global_step=epoch)

        self.min_high_writer.add_scalar(self.label + label, best_loss - min_delta, global_step=epoch)
        self.min_low_writer.add_scalar(self.label + label, best_loss - min_delta, global_step=epoch)

        self.loss_writer.add_scalar(self.label + label, loss, global_step=epoch)
        self.best_loss_writer.add_scalar(self.label + label, best_loss, global_step=epoch)
        if best_loss_2:
            self.best_loss_2_writer.add_scalar(self.label + label, best_loss_2, global_step=epoch)

    def __call__(self, train_loss, val_loss, epoch=None):
        epoch = epoch if epoch is not None else self.epoch
        if len(self.val_loss_history):
            train_loss = progressive_smooth(self.train_loss_history[-1], self.smoothing, train_loss)
            val_loss = progressive_smooth(self.val_loss_history[-1], self.smoothing, val_loss)
        self.train_loss_history = [*self.train_loss_history, train_loss][-self.history_length:]
        self.val_loss_history = [*self.val_loss_history, val_loss][-self.history_length:]
        mid_val_loss, min_delta_val_percent = self.calculate_interval(self.val_loss_history)
        mid_train_loss, min_delta_train_percent = self.calculate_interval(self.train_loss_history)
        min_delta_val = max(self.min_delta_val, min_delta_val_percent)
        min_delta_train = max(self.min_delta_train, min_delta_train_percent)

        self.log_stop(
            label="val_stop", epoch=epoch,
            loss=val_loss,
            min_delta=self.min_delta_val, min_delta_percent=min_delta_val_percent,
            best_loss=self.best_val_loss, best_loss_2=self.best_val_loss_2
        )
        self.log_stop(
            label="train_stop", epoch=epoch,
            loss=train_loss,
            min_delta=self.min_delta_train, min_delta_percent=min_delta_train_percent,
            best_loss=self.best_train_loss
        )

        if self.wait_counter < self.wait:
            self.wait_counter += 1
            if self.debug >= 3:
                print(f"INFO: Early stopping wait {self.wait_counter}/{self.wait}")
        elif not self.train_below_val and val_loss < train_loss and self.wait_train_below_val_counter < self.wait_train_below_val:
            self.wait_train_below_val_counter += 1
            if self.debug >= 3:
                print(f"INFO: Early stopping wait train below val {self.wait_train_below_val_counter}/{self.wait_train_below_val}")
        else:
            self.train_below_val = True
            if self.best_val_loss is None:
                self.update_best_2(val_loss)
                self.update_best(train_loss, val_loss)
            else:
                delta_val_loss = val_loss - self.best_val_loss
                delta_train_loss = train_loss - self.best_train_loss
                if self.debug >= 3:
                    print("INFO: Early stopping check", train_loss, val_loss, delta_train_loss, delta_val_loss, min_delta_val)
                rise = delta_val_loss > min_delta_val
                if rise:
                    self.rise_counter += 1
                    self.still_counter = 0  # It will need time to go down
                    if self.debug >= 2:
                        print(f"INFO: Early stopping rise {self.rise_counter}/{self.rise_patience}")
                    if self.rise_counter >= self.rise_patience:
                        self.early_stop("rise")
                else:
                    self.rise_counter = 0
                    still = abs(delta_val_loss) < min_delta_val
                    if still:
                        still_increment = 0
                        if self.min_min_delta_val < min_delta_val:
                            still_increment = 1.0/3
                        elif val_loss < self.best_val_loss_2:
                            still_increment = 0.5
                            self.update_best_2(val_loss)
                        elif delta_train_loss < -min_delta_train:
                            still_increment = 0.75
                        else:
                            still_increment = 1
                        self.still_counter += still_increment
                        if self.debug >= 2:
                            print(f"INFO: Early stopping still {self.still_counter}/{self.still_patience}")
                        if self.still_counter >= self.still_patience:
                            self.early_stop("still")
                    else:
                        self.update_best(train_loss, val_loss)
                        self.still_counter = 0



        self.still_writer.add_scalar(self.label + "patience", self.still_counter/self.still_patience, global_step=epoch)
        self.rise_writer.add_scalar(self.label + "patience", self.rise_counter/self.rise_patience, global_step=epoch)

        self.epoch = epoch + 1
        return self.early_stopped

    def update_state(self):
        self.best_state = self.model.state_dict()

    def update_best_2(self, val_loss):
        self.best_val_loss_2 = val_loss
        self.update_state()

    def update_best(self, train_loss, val_loss):
        self.best_train_loss = train_loss
        self.best_val_loss = val_loss
        if val_loss < self.best_val_loss_2:
            self.update_best_2(val_loss)

    def early_stop(self, reason="idk"):
        self.model.load_state_dict(self.best_state)
        self.early_stopped = True
        if self.debug >= 1:
            print(f"INFO: Early stopping due to {reason}")
