class TrainConfig:
    def __init__(self):
        self.data_path = "C:\Repos\solvro-challenge\data/"
        self.train_data_path = "train_data.txt"
        self.truth_file_path = "truth.txt"
        self.save_model_path = "best_model.pt"
        self.batch_size = 32
        self.epochs = 10
        self.tpu = True
        self.gpu = True
        self.wandb = True
        self.wandb_project = "andi"
        self.wandb_run_name = "simple_BiLSTM_CNN"
        self.entity = "blaku01"
        self.optimizer = "adam"
        self.lr = 0.01
        self.clipvalue = 0.5
        self.scheduler = "stepLR"
        self.gamma = 0.1
        self.step_size = 5
        self.num_workers = 4
        self.devices = 1


class ModelConfig:
    def __init__(self):
        self.input_size = 2
        self.hidden_size = 32
        self.num_classes = 5


class DataConfig:
    def __init__(self):
        self.max_length = 1000


class Config:
    def __init__(self):
        self.train = TrainConfig()
        self.model = ModelConfig()
        self.data = DataConfig()
