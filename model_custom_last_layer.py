import numpy as np
from tqdm import tqdm
import torch
from torch import nn, stack, cat
from transformers import BertForQuestionAnswering, BertTokenizer, AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
path_to_model = "xlnet-base-cased"
num_epoch = 3
batch_size = 8


class CustomROBERTAModel(nn.Module):
    def __init__(self):
        super(CustomROBERTAModel, self).__init__()
        self.model = AutoModelForQuestionAnswering.from_pretrained("xlnet-base-cased")
        ### New layers:
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 1)

    def forward(self, ids, mask):
        roberta_output = self.model(
            ids,
            attention_mask=mask,
            output_hidden_states=True)

        linear1_output = self.linear1(roberta_output.hidden_states[-1])
        linear2_output = torch.squeeze(self.linear2(linear1_output))

        return torch.mean(linear2_output)


def tokenize_function(examples):
    return tokenizer(examples["context"], padding="max_length", truncation=True)


def train(data_files, path_to_model):
    column_names = ['context', 'question', 'label', 'answer_starts']
    raw_dataset = load_dataset("csv", data_files=data_files, column_names=column_names)

    tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))
    full_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
    full_eval_dataset = tokenized_datasets["test"]
    full_train_dataloader = DataLoader(full_train_dataset, shuffle=True, batch_size=batch_size, collate_fn=lambda x: x)
    full_eval_dataloader = DataLoader(full_eval_dataset, batch_size=batch_size, collate_fn=lambda x: x)

    model = CustomROBERTAModel()  # You can pass the parameters if required to have more flexible model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  ## can be gpu ## TODO : change from cpu
    criterion = nn.MSELoss()  ## If required define your own criterion
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    model.train(True)
    running_loss = []
    num_examples = 0
    for epoch in range(num_epoch):
        for batch in tqdm(full_train_dataloader):  ## If you have a DataLoader()  object to get the data.
            if len(batch) < batch_size:
                continue
            num_examples += batch_size
            context = [b['context'] for b in batch]
            question = [b['question'] for b in batch]
            targets = torch.tensor(
                [float(b['label']) for b in batch])  ## assuming that data loader returns a tuple of data and
            # its targets
            optimizer.zero_grad()
            encoding_c = tokenizer.batch_encode_plus(context, return_tensors='pt', padding=True, truncation=True,
                                                     max_length=50, add_special_tokens=True)
            encoding_q = tokenizer.batch_encode_plus(question, return_tensors='pt', padding=True, truncation=True,
                                                     max_length=50, add_special_tokens=True)
            input_ids = cat((encoding_c['input_ids'], encoding_q['input_ids']), 1)
            attention_mask = cat((encoding_c['attention_mask'], encoding_q['attention_mask']), 1)
            # stacked_input_ids = stack((input_ids,input_ids),-1)
            outputs = model(input_ids, mask=attention_mask)
            # outputs = F.log_softmax(outputs, dim=1)
            # outputs = torch.tensorflow
            outputs = outputs.type(torch.FloatTensor)
            targets = targets.type(torch.FloatTensor)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
        #     if num_examples >= 300:
        #         plt.close()
        #         plt.plot(range(len(running_loss)), running_loss, 'ro-')
        #         plt.show()
        #         num_examples = 0
        # plt.savefig('training_error.png')
    torch.save(model.state_dict(), path_to_model)


def test(data_files, path_to_model=None):
    column_names = ['context', 'question', 'label', 'answer_starts']
    raw_dataset = load_dataset("csv", data_files=data_files, column_names=column_names)

    tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))
    full_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)
    small_eval_dataloader = DataLoader(small_eval_dataset, shuffle=True, batch_size=batch_size)
    full_eval_dataloader = DataLoader(full_eval_dataset, batch_size=batch_size)

    model = CustomROBERTAModel()
    if path_to_model is None:
        train(data_files, path_to_model)
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    loss = []
    results = []
    for batch in tqdm(full_eval_dataloader):  ## If you have a DataLoader()  object to get the data.
        if len(batch) < batch_size:
            continue
        context = [b['context'] for b in batch]
        question = [b['question'] for b in batch]
        targets = torch.tensor(
            [float(b['label']) for b in batch])  ## assuming that data loader returns a tuple of data and
        # its targets

        encoding_c = tokenizer.batch_encode_plus(context, return_tensors='pt', padding=True, truncation=True,
                                                 max_length=50, add_special_tokens=True)
        encoding_q = tokenizer.batch_encode_plus(question, return_tensors='pt', padding=True, truncation=True,
                                                 max_length=50, add_special_tokens=True)
        input_ids = cat((encoding_c['input_ids'], encoding_q['input_ids']), 1)
        attention_mask = cat((encoding_c['attention_mask'], encoding_q['attention_mask']), 1)
        # stacked_input_ids = stack((input_ids,input_ids),-1)
        outputs = model(input_ids, mask=attention_mask)
        # outputs = F.log_softmax(outputs, dim=1)
        # outputs = torch.tensorflow
        outputs = torch.squeeze(outputs.type(torch.FloatTensor))
        targets = targets.type(torch.FloatTensor)
        loss_f = nn.MSELoss()
        loss.append(loss_f(outputs, targets).tolist())
        results.extend([outputs, targets])
    loss_np = np.array(loss)
    return np.mean(loss_np), np.count_nonzero(loss_np < 0.1) / loss_np.size, results


if __name__ == "__main__":
    data_files = {'train': 'data2/train.csv',
                  'test': 'data2/test.csv'}
    train(data_files, path_to_model)
    test_error = test(data_files, path_to_model)
    print(f"********* TEST ERROR ******\n"
          f"\t{test_error[0]}, while accuracy is {test_error[1]}")
    print(test_error[2])
