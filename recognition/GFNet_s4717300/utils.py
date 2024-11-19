class Environment():
    """
    Container for hyperparameter and other miscellaneous
    """
    def __init__(self) -> None:
        # Hyperparameter
        self.learning_rate  = 0.001 
        self.weight_decay   = 0.001
        self.dropout        = 0.0
        self.drop_path      = 0.1       # Probability of dropping an entire network path from an iteration
        self.batch_size     = 32
        self.patch_size     = 16        # Size of the chunks to split up the image in pixels
        self.embed_dim      = 783 
        self.depth          = 12         # Number of global filter layers to use in the network
        self.ff_ratio       = 3         # Ratio of input to hidden layer size in the classification feed forward network
        self.epochs         = 1

        # Model Metadata
        self.dataset_path   = None      # Path of training and testing data
        self.model_path     = None      # Path of a model checkpoint to be loaded
        self.save_check     = False     # True if use wants to save checkpoints every 10 epochs
        self.monitor        = False     # Outputs estimated loss values every epoch
        self.tag            = "GFNet"   # Name of the model, will be appended to all output

        # Dataset Metadata
        self.n_classes      = 2         # WARNING: Must change this value manually if performing inference

        # Data collection
        self.training_losses            = []
        self.estimated_test_losses      = []
        self.test_losses                = []
        self.estimated_test_accuracy    = []
        self.test_accuracy              = []

