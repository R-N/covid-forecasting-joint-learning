from .util import progressive_smooth
import scipy.stats as st
from math import sqrt
from torch.utils.tensorboard import SummaryWriter


class EarlyStopping:
    def __init__(
        self,
        model,
        wait=50, wait_train_below_val=20,
        rise_patience=20, still_patience=12,
        interval_percent=0.05,
        min_delta_val_percent=0.15, min_delta_train_percent=0.025,
        history_length=None,
        smoothing=0.25,
        interval_mode=2,
        max_epoch=100,
        max_nan=None,
        rise_forgiveness=0.6,
        still_forgiveness=0.6,
        mini_forgiveness_mul=0.1,
        wait_forgive_count=1,
        rel_val_reduction_still_tolerance=0.1,
        val_reduction_still_tolerance=0.35,
        train_reduction_still_tolerance=0.25,
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
        self.rise_patience = rise_patience
        self.still_patience = still_patience
        self.min_delta_val = 0
        self.min_delta_train = 0
        self.min_delta_val_percent = min_delta_val_percent
        self.min_delta_train_percent = min_delta_train_percent
        self.rise_counter = 0
        self.still_counter = 0
        self.history_length = history_length or min(rise_patience, still_patience)
        self.train_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = None
        self.best_train_loss = None
        self.best_val_loss_2 = None
        self.stopped = False
        self.smoothing = min(1.0, max(0, smoothing))
        self.interval_percent = interval_percent
        self.debug = debug
        self.interval_funcs = [
            self.calculate_interval_0,
            self.calculate_interval_1,
            self.calculate_interval_2
        ]
        self.interval_mode = interval_mode

        self.rise_forgiveness = rise_forgiveness
        self.still_forgiveness = still_forgiveness
        self.mini_forgiveness_mul = mini_forgiveness_mul
        self.wait_forgive_count = wait_forgive_count
        self.rel_val_reduction_still_tolerance = rel_val_reduction_still_tolerance
        self.val_reduction_still_tolerance = val_reduction_still_tolerance
        self.train_reduction_still_tolerance = train_reduction_still_tolerance

        self.max_epoch = max_epoch
        self.log_dir = log_dir
        self.label = label
        self.epoch = 0
        self.active = False

        self.max_nan = max_nan or int(0.5 * (self.wait - self.history_length))
        self.nan_counter = 0

        if self.log_dir is not None:
            assert self.label is not None

            self.min_percent_high_writer = SummaryWriter(log_dir + "/min_percent_high")
            self.min_percent_low_writer = SummaryWriter(log_dir + "/min_percent_low")

            self.min_high_writer = SummaryWriter(log_dir + "/min_high")
            self.min_low_writer = SummaryWriter(log_dir + "/min_low")

            self.loss_writer = SummaryWriter(log_dir + "/loss")
            self.best_loss_writer = SummaryWriter(log_dir + "/best_loss")
            self.best_loss_2_writer = SummaryWriter(log_dir + "/best_loss_2")

            self.still_writer = SummaryWriter(log_dir + "/still")
            self.rise_writer = SummaryWriter(log_dir + "/rise")

    def calculate_interval(self, *args, **kwargs):
        return self.interval_funcs[self.interval_mode](*args, **kwargs)

    def calculate_interval_0(self, val=True):
        history = self.val_loss_history if val else self.train_loss_history
        percent = self.min_delta_val_percent if val else self.min_delta_train_percent
        max_val = max(history)
        delta = percent * max_val
        mid = history[-2]
        return mid, delta

    def calculate_interval_1(self, val=True):
        history = self.val_loss_history if val else self.train_loss_history
        min_val = min(history)
        max_val = max(history)
        delta = 0.5 * (1.0 - self.interval_percent) * (max_val - min_val)
        mid = 0.5 * (min_val + max_val)
        return mid, delta

    def calculate_interval_2(self, val=True):
        history = self.val_loss_history if val else self.train_loss_history
        mean = sum(history) / self.history_length
        sum_err = sum([(mean - x)**2 for x in history])
        stdev = sqrt(1 / (self.history_length - 2) * sum_err)
        mul = st.norm.ppf(1.0 - self.interval_percent) if self.interval_percent >= 0 else 2 + self.interval_percent
        sigma = mul * stdev
        return mean, sigma

    def log_stop(
        self,
        label, epoch,
        loss,
        min_delta=None, min_delta_percent=None,
        best_loss=None, best_loss_2=None
    ):
        self.loss_writer.add_scalar(self.label + label, loss, global_step=epoch)
        self.loss_writer.flush()

        if min_delta is not None:
            self.min_high_writer.add_scalar(self.label + label, best_loss + min_delta, global_step=epoch)
            self.min_low_writer.add_scalar(self.label + label, best_loss - min_delta, global_step=epoch)
            self.min_high_writer.flush()
            self.min_low_writer.flush()

        if min_delta_percent is not None:
            self.min_percent_high_writer.add_scalar(self.label + label, best_loss + min_delta_percent, global_step=epoch)
            self.min_percent_low_writer.add_scalar(self.label + label, best_loss - min_delta_percent, global_step=epoch)
            self.min_percent_high_writer.flush()
            self.min_percent_low_writer.flush()

        if best_loss is not None:
            self.best_loss_writer.add_scalar(self.label + label, best_loss, global_step=epoch)
            self.best_loss_writer.flush()

        if best_loss_2 is not None:
            self.best_loss_2_writer.add_scalar(self.label + label, best_loss_2, global_step=epoch)
            self.best_loss_2_writer.flush()

    def __call__(self, train_loss, val_loss, epoch=None):
        epoch = epoch if epoch is not None else self.epoch
        if len(self.val_loss_history):
            train_loss = progressive_smooth(self.train_loss_history[-1], self.smoothing, train_loss)
            val_loss = progressive_smooth(self.val_loss_history[-1], self.smoothing, val_loss)
        self.train_loss_history = [*self.train_loss_history, train_loss][-self.history_length:]
        self.val_loss_history = [*self.val_loss_history, val_loss][-self.history_length:]

        if self.wait_counter < self.wait:
            self.wait_counter += 1
        elif not self.active and val_loss < train_loss and self.wait_train_below_val_counter < self.wait_train_below_val:
            self.wait_train_below_val_counter += 1
        elif not self.active:
            self.active = True
            for i in range(self.wait_forgive_count):
                self.forgive_still()
                self.forgive_rise()
            print(f"INFO: Early stopping active at epoch {epoch} after skipping {self.nan_counter}/{self.max_nan} NaN epochs and waiting {self.wait_train_below_val_counter}/{self.wait_train_below_val} epochs for train to get below val")

        if self.best_val_loss is None:
            self.update_best_val_2(val_loss)
            self.update_best_train(train_loss)
            self.update_best_val(val_loss)
            self.recalculate_delta_val()
            self.recalculate_delta_train()

        min_delta_val = self.min_delta_val
        min_delta_train = self.min_delta_train

        if self.log_dir:
            self.log_stop(
                label="val_stop", epoch=epoch,
                loss=val_loss,
                min_delta=self.min_delta_val,
                best_loss=self.best_val_loss, best_loss_2=self.best_val_loss_2
            )
            self.log_stop(
                label="train_stop", epoch=epoch,
                loss=train_loss,
                min_delta=self.min_delta_train,
                best_loss=self.best_train_loss
            )

        delta_val_loss = val_loss - self.best_val_loss
        delta_train_loss = train_loss - self.best_train_loss

        train_fall = delta_train_loss < -min_delta_train
        if train_fall:
            self.update_best_train(train_loss)
        if delta_train_loss < min_delta_train:
            self.recalculate_delta_train()

        rise = delta_val_loss > min_delta_val
        if rise:
            self.rise_counter += 1
            self.forgive_still(self.mini_forgiveness_mul)  # It will need time to go down
            if self.rise_counter >= self.rise_patience:
                self.early_stop("rise", epoch)
        else:
            still = abs(delta_val_loss) < min_delta_val
            self.recalculate_delta_val()
            if still:
                self.forgive_rise(self.mini_forgiveness_mul)
                still_increment = 1
                if val_loss < self.val_loss_history[-1]:
                    still_increment *= (1.0 - self.rel_val_reduction_still_tolerance)
                if val_loss < self.best_val_loss_2:
                    still_increment *= (1.0 - self.val_reduction_still_tolerance)
                    self.update_best_val_2(val_loss)
                if train_fall:
                    still_increment *= (1.0 - self.train_reduction_still_tolerance)
                self.still_counter += still_increment
                if self.still_counter >= self.still_patience:
                    self.early_stop("still", epoch)
            else:
                self.update_best_val(val_loss)
                self.forgive_rise()
                self.forgive_still()

        self.rise_counter = max(0, min(self.rise_patience, self.rise_counter))
        self.still_counter = max(0, min(self.still_patience, self.still_counter))
        still_percent = self.still_counter / self.still_patience
        rise_percent = self.rise_counter / self.rise_patience

        if self.log_dir:
            self.still_writer.add_scalar(self.label + "patience", still_percent, global_step=epoch)
            self.rise_writer.add_scalar(self.label + "patience", rise_percent, global_step=epoch)

            self.still_writer.flush()
            self.rise_writer.flush()

        # stilling = still_percent >= (1.0 - self.still_forgiveness)
        # rising = rise_percent >= (1.0 - self.rise_forgiveness)

        if self.max_epoch and epoch >= self.max_epoch and (rise or still):
            self.stop()
            if self.debug >= 1:
                print(f"INFO: Stopping at max epoch {epoch}")

        self.epoch = epoch + 1
        return self.stopped

    def recalculate_delta_val(self):
        mid_val_loss, self.min_delta_val = self.calculate_interval(val=True)
        if self.best_val_loss_2 - self.best_val_loss < -self.min_delta_val:
            self.best_val_loss = self.best_val_loss_2

    def recalculate_delta_train(self):
        mid_train_loss, self.min_delta_train = self.calculate_interval(val=False)

    def update_state(self):
        self.best_state = self.model.state_dict()

    def update_best_val_2(self, val_loss):
        self.best_val_loss_2 = val_loss
        self.update_state()

    def update_best_train(self, train_loss):
        self.best_train_loss = train_loss

    def update_best_val(self, val_loss):
        self.best_val_loss = val_loss
        if val_loss < self.best_val_loss_2:
            self.update_best_val_2(val_loss)

    def stop(self):
        if not self.stopped:
            self.model.load_state_dict(self.best_state)
            self.stopped = True

    def early_stop(self, reason="idk", epoch=None):
        if not self.active:
            self.forgive_wait()
            return
        epoch = epoch if epoch is not None else self.epoch
        self.stop()
        if self.debug >= 1:
            print(f"INFO: Early stopping due to {reason} at epoch {epoch}")

    def calculate_forgiveness(self, counter, forgiveness, patience):
        return max(0, counter - forgiveness * patience)

    def forgive_rise(self, mul=1):
        self.rise_counter = self.calculate_forgiveness(self.rise_counter, mul * self.rise_forgiveness, self.rise_counter)

    def forgive_still(self, mul=1):
        self.still_counter = self.calculate_forgiveness(self.still_counter, mul * self.still_forgiveness, self.still_counter)

    def forgive_wait(self):
        if self.debug >= 2:
            print(f"INFO: Early stopping forgiven due to wait")

    def step_nan(self):
        if self.nan_counter < self.max_nan:
            self.nan_counter += 1
            if self.wait_counter < self.wait:
                self.wait_counter += 1
            return True
        return False
