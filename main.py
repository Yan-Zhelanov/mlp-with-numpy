from configs.experiment_config import ExperimentConfig
from executors.trainer import Trainer

if __name__ == '__main__':
    trainer = Trainer(ExperimentConfig())

    # one batch overfitting
    # trainer.batch_overfit()

    # model training
    trainer.fit()
