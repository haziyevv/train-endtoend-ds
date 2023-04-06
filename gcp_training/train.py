
import re
import os
import logging

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from google.cloud import bigquery
from google.cloud import storage

from .data import create_data_loader
from .model import AzeNewsModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


bq_client = bigquery.Client(project=os.getenv("CLOUD_ML_PROJECT_ID"))

# to query the data from BigQuery and select texts and categories
query = f"SELECT DISTINCT text, category FROM `azenews.news`"
query_job = bq_client.query(query)
result = query_job.result().to_dataframe()

# to clean the texts 
texts = [
    re.sub(r"\[.*?\]|[\s\u200b]+", " ", x).replace("  ", "").strip().lower()
    for x in result.text.values
]

# to get the categories
categ2index = {"iqtisadi": 0, "medeniyyet": 1, "siyasi": 2, "idman": 3, "ikt": 4}
categories = [categ2index[x] for x in result.category.values]


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client(project="CLOUD_ML_PROJECT_ID")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print("File {} uploaded to {}.".format(source_file_name, destination_blob_name))


tokenizer = AutoTokenizer.from_pretrained("heziyevv/aze-bert-tokenizer-large")
model = AzeNewsModel(tokenizer.vocab_size, 128)
dataloader = create_data_loader(texts, categories, 512, 32, tokenizer)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(100):
    total_loss = 0.0
    for batch in dataloader:
        input_ids, attention_mask, category = batch
        y_pred = model(input_ids)
        loss = loss_fn(y_pred, torch.tensor(category))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    if total_loss < 0.1:
        break
    logger.info(f"Epoch {epoch}: loss: {total_loss}")

torch.save(model.state_dict(), "aze-ds-model-first")
upload_blob("xeber-bucket", "aze-ds-model-first", "models/aze-ds-model-first")
