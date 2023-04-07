import re
import os
import logging
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from google.cloud import bigquery
from sklearn.model_selection import train_test_split

from .data import create_data_loader
from .model import AzeNewsModel
from .utils import upload_blob

# use tensorboard to visualize the training process


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class NewsClassifer:
    def __init__(
        self,
        bq_project_id,
        tokenizer_name,
        model_name,
        max_epochs=1000,
        lr=1e-4,
        batch_size=256,
        eval_batch_size=64,
        max_num_tokens=256,
        bucket_name="azenews",
    ):
        self.bq_client = bigquery.Client(project=os.getenv(bq_project_id))
        self.categ2index = {
            "iqtisadi": 0,
            "medeniyyet": 1,
            "siyasi": 2,
            "idman": 3,
            "ikt": 4,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AzeNewsModel(vocab_size=self.tokenizer.vocab_size, emb_size=128)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_num_tokens = max_num_tokens

        self.model_name = model_name
        self.bucket_name = bucket_name

    def preprocess(self, texts):
        """Preprocess the texts by removing the brackets and extra spaces and useless informations"""
        return [
            re.sub(r"\[.*?\]|[\s\u200b]+", " ", x)
            .replace("  ", "")
            .strip()
            .lower()
            .replace("bizim telegram kanalımıza abunə olun", "")
            for x in texts
        ]

    def get_data_from_bq(self, test_size=0.1, random_state=42):
        """Get the data from BigQuery and split it into train and eval sets"""
        query = f"SELECT DISTINCT text, category FROM `azenews.news`"
        query_job = self.bq_client.query(query)
        result = query_job.result().to_dataframe()

        train_set, eval_set = train_test_split(
            result, test_size=test_size, random_state=random_state
        )

        # preprocess the train and eval sets
        train_texts = self.preprocess(train_set.text.values)
        eval_texts = self.preprocess(eval_set.text.values)

        # convert the categories to indices
        train_categories = [self.categ2index[x] for x in train_set.category.values]
        eval_categories = [self.categ2index[x] for x in eval_set.category.values]

        # create the dataloaders
        train_dataloader = create_data_loader(
            texts=train_texts,
            categories=train_categories,
            max_len=self.max_num_tokens,
            batch_size=self.batch_size,
            tokenizer=self.tokenizer,
            shuffle=True,
        )
        eval_dataloader = create_data_loader(
            texts=eval_texts,
            categories=eval_categories,
            max_len=self.max_num_tokens,
            batch_size=self.eval_batch_size,
            tokenizer=self.tokenizer,
            shuffle=False,
        )

        return train_dataloader, eval_dataloader

    def train_epoch(self, train_dataloader):
        """Train the model for one epoch"""
        total_loss = 0.0
        for batch in train_dataloader:
            input_ids, _, target_category = batch
            y_pred = self.model(input_ids)

            loss = self.loss_fn(y_pred, torch.tensor(target_category))
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()

        return total_loss

    def eval_epoch(self, eval_dataloader):
        """Evaluate the model for one epoch"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids, _, target_category = batch
                y_pred = self.model(input_ids)
                loss = self.loss_fn(y_pred, torch.tensor(target_category))
                total_loss += loss.item()

        self.model.train()

        return total_loss

    def train_loop(self, train_dataloader, eval_dataloader):
        writer = SummaryWriter("logs")

        # apply early stopping to stop the training process if the eval loss is not decreasing
        tol = 1e-4 # tolerance that we allow the eval loss to improve at least in each pass
        early_stopping_counter = 0
        previous_loss = float("inf")

        for epoch in range(self.max_epochs):
            train_loss = self.train_epoch(train_dataloader)

            if epoch % 10 == 0: # evaluate the model every 10 epochs
                eval_loss = self.eval_epoch(eval_dataloader)

                if previous_loss - eval_loss < tol:
                    early_stopping_counter += 1
                else:
                    early_stopping_counter = 0

                if early_stopping_counter > 3:
                    break

                previous_loss = eval_loss
                logger.info(f"Epoch {epoch}: eval loss: {eval_loss}")
                writer.add_scalar("validation_loss", eval_loss, epoch)
            writer.add_scalar("training_loss", train_loss, epoch)
            logger.info(f"Epoch {epoch}: loss: {train_loss}")
        writer.close()

    def train(self):
        """Train the model and save it and upload it to the google cloud storage bucket"""
        train_dataloader, eval_dataloader = self.get_data_from_bq()
        self.train_loop(train_dataloader, eval_dataloader)
        torch.save(self.model.state_dict(), self.model_name)
        upload_blob(self.bucket_name, self.model_name, f"models/{self.model_name}")
