class Hyperparameters:
    def __init__(self):
        self.data_size_train = 1024
        self.data_size_test = 1024
        self.num_epochs = 15
        self.batch_size = 128
        self.learning_rate = 0.01
        self.num_classes = 10

    def print_hyperparameters(self):
        print("Hyperparameters:")
        print("data_size_train:", self.data_size_train)
        print("data_size_test:", self.data_size_test)
        print("num_epochs:", self.num_epochs)
        print("batch_size:", self.batch_size)
        print("learning_rate:", self.learning_rate)
        print("num_classes:", self.num_classes)
