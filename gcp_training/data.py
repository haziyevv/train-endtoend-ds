import torch
from torch.utils.data import Dataset, DataLoader

class NewsDataset(Dataset):
    def __init__(self, texts, categories, max_length, tokenizer):
        self.texts = texts
        self.categories = categories
        self.max_len = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        category = self.categories[idx]

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
        )
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        return input_ids, attention_mask, category

def collate_fn(batch):
    input_ids, attention_mask, categories = zip(*batch)
    return torch.stack(input_ids), torch.stack(attention_mask), list(categories)



def create_data_loader(texts, categories, max_len, batch_size, tokenizer):
    dataset = NewsDataset(texts, categories, max_len, tokenizer)

    # create a dataloader that pads the sequences to the maximum length
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
