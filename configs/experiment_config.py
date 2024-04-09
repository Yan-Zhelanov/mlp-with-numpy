import os

from easydict import EasyDict

from configs.data_config import data_config
from configs.mlp_config import model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

experiment_cfg = EasyDict()
experiment_cfg.seed = 0
experiment_cfg.num_epochs = 10

# Train parameters
experiment_cfg.train = EasyDict()
experiment_cfg.train.batch_size = 64
experiment_cfg.train.learning_rate = 5e-2

# Overfit parameters
experiment_cfg.overfit = EasyDict()
experiment_cfg.overfit.num_iterations = 500

# Neptune parameters
experiment_cfg.neptune = EasyDict()
experiment_cfg.neptune.env_path = os.path.join(ROOT_DIR, '.env')
experiment_cfg.neptune.project = None
experiment_cfg.neptune.experiment_name = None
experiment_cfg.neptune.run_id = None
experiment_cfg.neptune.dependencies_path = os.path.join(ROOT_DIR, 'requirements.txt')

# Checkpoints parameters
experiment_cfg.checkpoints_dir = os.path.join(ROOT_DIR, 'checkpoints', experiment_cfg.neptune.experiment_name)

# Data parameters
experiment_cfg.data = data_config
experiment_cfg.data.num_classes = 7

# Model parameters
experiment_cfg.model = model_cfg
