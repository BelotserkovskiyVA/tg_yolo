
class EpochsLogger():
    
    def __init__(self):
        self.epochs_number = 19
        self.current_epoch = 0
        self.save_dir = ''
    
    def set_epochs(self, epochs):
        self.epochs_number = epochs
    
    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

epochs_logger = EpochsLogger()

#351 # epochs_logger.set_current_epoch(epoch)
#352 # epochs_logger.set_epochs(epochs - 1)

#421 # epochs_logger.set_current_epoch(epochs)
#422 # epochs_logger.set_epochs(epochs - 1)

#423 # epochs_logger.save_dir = save_dir
