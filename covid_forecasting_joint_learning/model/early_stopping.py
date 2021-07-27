class EarlyStopping:
    def __init__(self, model, rise_patience=5, still_patience=10, min_delta=1e-5, debug=False):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.model = model
        self.rise_patience = rise_patience
        self.still_patience = still_patience
        self.min_delta = min_delta
        self.rise_counter = 0
        self.still_counter = 0
        self.best_val_loss = None
        self.best_train_loss = None
        self.early_stopped = False
        self.debug = debug

    def __call__(self, train_loss, val_loss):
        if self.best_val_loss == None:
            self.update_best(train_loss, val_loss)
        else:
            delta_val_loss = self.best_val_loss - val_loss
            rise = delta_val_loss < self.min_delta
            if rise:
                self.rise_counter += 1
                if self.debug:
                    print(f"INFO: Early stopping rise {self.rise_counter}/{self.rise_patience}")
                if self.rise_counter >= self.rise_patience:
                    self.early_stop("rise")
            else:
                self.rise_counter = 0
                still = abs(delta_val_loss) < self.min_delta
                if val_loss < self.best_val_loss or (still and train_loss < self.best_train_loss):
                    self.update_best(train_loss, val_loss)
                if still:
                    self.still_counter += 1
                    if self.debug:
                        print(f"INFO: Early stopping still {self.still_counter}/{self.still_patience}")
                    if self.still_counter >= self.still_patience:
                        self.early_stop("still")
                else:
                    self.still_counter = 0
        return self.early_stopped

    def update_best(self, train_loss, val_loss):
        self.best_train_loss = train_loss
        self.best_val_loss = val_loss
        self.best_state = self.model.state_dict()

    def early_stop(self, reason="idk"):
        self.model.load_state_dict(self.best_state)
        self.early_stopped = True
        if self.debug:
            print(f"INFO: Early stopping due to {reason}")
