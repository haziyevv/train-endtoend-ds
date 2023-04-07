import os
import yaml

from .train import NewsClassifer

# read the config file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":
    trainer = NewsClassifer(
        bq_project_id=os.getenv("CLOUD_ML_PROJECT_ID"),
        tokenizer_name=config["tokenizer_name"],
        model_name=config["model_name"],
        max_epochs=int(config["max_epochs"]),
        lr=float(config["learning_rate"]),
        batch_size=int(config["batch_size"]),
        eval_batch_size=int(config["eval_batch_size"]),
        max_num_tokens=int(config["max_num_tokens"]),
        bucket_name=config["bucket_name"],
    )

    trainer.train()
