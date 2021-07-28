class EarlyStopping:
    def __init__(
        self, 
        model, 
        rise_patience=5, still_patience=10, 
        min_delta_val=1e-5, min_delta_val_percent=0.095, 
        min_delta_train=1.5e-6, min_delta_train_percent=0.020, 
        smoothing=5, 
        debug=0
    ):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.model = model
        self.rise_patience = rise_patience
        self.still_patience = still_patience
        self.min_delta_val = min_delta_val
        self.min_delta_val_percent = min_delta_val_percent
        self.min_delta_train = min_delta_train
        self.min_delta_train_percent = min_delta_train_percent
        self.rise_counter = 0
        self.still_counter = 0
        self.val_loss_history = []
        self.train_loss_history = []
        self.best_val_loss = None
        self.best_train_loss = None
        self.early_stopped = False
        self.smoothing = smoothing
        self.debug = debug

    def __call__(self, train_loss, val_loss):
        self.train_loss_history = [*self.train_loss_history, train_loss][-self.smoothing:]
        self.val_loss_history = [*self.val_loss_history, val_loss][-self.smoothing:]
        train_loss = sum(self.train_loss_history)/len(self.train_loss_history)
        val_loss = sum(self.val_loss_history)/len(self.val_loss_history)
        if self.best_val_loss is None:
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
                if self.debug >= 2:
                    print(f"INFO: Early stopping rise {self.rise_counter}/{self.rise_patience}")
                if self.rise_counter >= self.rise_patience:
                    self.early_stop("rise")
            else:
                self.rise_counter = 0
                still = abs(delta_val_loss) < min_delta_val
                min_delta_train_percent = self.min_delta_train_percent * train_loss
                min_delta_train = max(self.min_delta_train, min_delta_train_percent)
                still = not (val_loss < self.best_val_loss or (still and delta_train_loss < -min_delta_train))
                if still:
                    self.still_counter += 1
                    if self.debug >= 2:
                        print(f"INFO: Early stopping still {self.still_counter}/{self.still_patience}")
                    if self.still_counter >= self.still_patience:
                        self.early_stop("still")
                else:
                    self.update_best(train_loss, val_loss)
                    self.still_counter = 0
        return self.early_stopped

    def update_best(self, train_loss, val_loss):
        self.best_train_loss = train_loss
        self.best_val_loss = val_loss
        self.best_state = self.model.state_dict()

    def early_stop(self, reason="idk"):
        self.model.load_state_dict(self.best_state)
        self.early_stopped = True
        if self.debug >= 1:
            print(f"INFO: Early stopping due to {reason}")
