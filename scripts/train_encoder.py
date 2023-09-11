import argparse
import itertools
import json
import os
import time

import datasets
import torch
from datasets.load import load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel


class QueryRefEncoderMainModel(torch.nn.Module):
    """
    heavily borrowed from WebGLM: https://github.com/THUDM/WebGLM
    """

    def __init__(self, model_dir) -> None:
        super().__init__()
        self.question_encoder = AutoModel.from_pretrained(model_dir)
        self.reference_encoder = AutoModel.from_pretrained(model_dir)

        total = sum([param.nelement() for param in self.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def forward(self, question, pos, neg):
        global args

        q = self.question_encoder(**question)
        r_pos = self.reference_encoder(**pos)
        r_neg = self.reference_encoder(**neg)
        cls_q = self.mean_pooling(q[0], question["attention_mask"])
        cls_q /= args.temp
        cls_r_pos = self.mean_pooling(r_pos[0], pos["attention_mask"])
        cls_r_neg = self.mean_pooling(r_neg[0], neg["attention_mask"])

        method = "cos"

        if method == "inner_product":
            l_pos = torch.matmul(cls_q, torch.transpose(cls_r_pos, 0, 1))
            l_neg = torch.matmul(cls_q, torch.transpose(cls_r_neg, 0, 1))
        elif method == "cos":
            l_pos = torch.matmul(cls_q, torch.transpose(cls_r_pos, 0, 1)) / (cls_q.norm() * cls_r_pos.norm())
            l_neg = torch.matmul(cls_q, torch.transpose(cls_r_neg, 0, 1)) / (cls_q.norm() * cls_r_neg.norm())
        else:
            raise NotImplementedError

        return l_pos, l_neg

    @staticmethod
    def loss(l_pos, l_neg):
        return torch.nn.functional.cross_entropy(torch.cat([l_pos, l_neg], dim=1),
                                                 torch.arange(0, len(l_pos), dtype=torch.long, device=args.device))

    @staticmethod
    def num_correct(l_pos, l_neg):
        return ((torch.diag(l_pos) > torch.diag(l_neg)) == True).sum()

    @staticmethod
    def acc(l_pos, l_neg):
        return ((torch.diag(l_pos) > torch.diag(l_neg)) == True).sum() / len(l_pos)


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(WarmupLinearScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return (1 - self.ratio) * step / float(max(1, self.warmup))

        return max(
            0.0,
            1.0 + (self.ratio - 1) * (step - self.warmup) / float(max(1.0, self.total - self.warmup)),
        )


def move_dict_to_device(obj, device):
    for key in obj:
        obj[key] = obj[key].to(device)


def collate(data):
    question = tokenizer([item["question"] for item in data], return_tensors="pt", padding=True, truncation=True)
    positive_reference = tokenizer([item["positive_reference"] for item in data], return_tensors="pt", padding=True,
                                   truncation=True)
    negative_reference = tokenizer([item["negative_reference"] for item in data], return_tensors="pt", padding=True,
                                   truncation=True)

    for key in question: question[key] = question[key].to(args.device)
    for key in positive_reference: positive_reference[key] = positive_reference[key].to(args.device)
    for key in negative_reference: negative_reference[key] = negative_reference[key].to(args.device)

    return question, positive_reference, negative_reference


def eval():
    model.eval()
    with torch.no_grad():
        total_acc = 0
        for q, pos, neg in eval_loader:
            results = model(q, pos, neg)
            tot_cr = model.num_correct(*results)
            total_acc += tot_cr

        print("EVALUATION, Acc: %10.6f" % (total_acc / len(eval_set)))


def save(name):
    os.makedirs(log_dir, exist_ok=True)
    model.question_encoder.save_pretrained(os.path.join(log_dir, name, "query_encoder"))
    model.reference_encoder.save_pretrained(os.path.join(log_dir, name, "reference_encoder"))


def train(max_epoch=10, eval_step=200, save_step=400, print_step=50):
    step = 0
    for epoch in range(0, max_epoch):
        print("EPOCH %d" % epoch)
        for q, pos, neg in train_loader:
            model.train()
            step += 1
            opt.zero_grad()
            results = model(q, pos, neg)
            loss = model.loss(*results)

            if step % print_step == 0:
                print("Step %4d, Loss, Acc: %10.6f, %10.6f" % (step, loss, model.acc(*results)))

            loss.backward()
            opt.step()

            scheduler.step()
            model.zero_grad()
            if step % eval_step == 0:
                eval()
                pass
            if step % save_step == 0:
                save("step-%d" % (step))

        save("step-%d-epoch-%d" % (step, epoch))
        # eval()


def data_process():
    """Use preprocesed data """
    with open("../raw_data/retro_source_report.txt", "r") as file:
        document = file.read()
        documents = document.split("Doc ")
        documents = ["Doc " + document for document in documents[1:]]

    with open("../raw_data/retro_qa.json", "r") as file:
        data = json.load(file)

    features = []
    for item in data:
        print("item: ", item)
        positive_ids = [int(i) for i in item["pos_index"].split(",")]
        negative_ids = [int(i) for i in item["neg_index"].split(",")]
        for pos, neg in itertools.product(positive_ids, negative_ids):
            print(len(features), pos, neg)
            features.append({
                "question": item["input"],
                "positive_label": pos,
                "positive_reference": documents[pos],
                "negative_label": neg,
                "negative_reference": documents[neg]
            })
    num_training = 5 * int(len(data) * 0.8)
    train_data = features[:num_training]
    eval_data = features[num_training:]
    train_data = datasets.Dataset.from_list(train_data)
    eval_data = datasets.Dataset.from_list(eval_data)
    train_data.save_to_disk("../raw_data/retriever/train")
    eval_data.save_to_disk("../raw_data/retriever/eval")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--max_epoch", type=int, default=3)
    args.add_argument("--eval_step", type=int, default=40)
    args.add_argument("--save_step", type=int, default=40)
    args.add_argument("--print_step", type=int, default=40)
    args.add_argument("--device", type=str, default="cpu")
    args.add_argument("--temp", type=float, default=0.05)
    args.add_argument("--train_batch_size", type=int, default=64)
    args.add_argument("--eval_batch_size", type=int, default=32)
    args.add_argument("--lr", type=float, default=1e-6)
    args.add_argument("--warmup", type=int, default=100)
    args.add_argument("--total", type=int, default=1000)
    args.add_argument("--ratio", type=float, default=0.0)
    args.add_argument("--save_dir", type=str, default="./retriever_runs")
    args.add_argument("--train_data_dir", type=str, default="../raw_data/retriever")
    args.add_argument("--train_data_dir", type=str, default="m3e-small")

    args = args.parse_args()

    log_dir = os.path.join(args.save_dir, time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time())))

    train_set = load_from_disk(os.path.join(args.train_data_dir, "train"))
    eval_set = load_from_disk(os.path.join(args.train_data_dir, "eval"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, collate_fn=collate)
    eval_loader = DataLoader(eval_set, batch_size=args.eval_batch_size, collate_fn=collate)

    model = QueryRefEncoderMainModel(args.model_dir)
    model = model.to(args.device)
    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    scheduler_args = {
        "warmup": args.warmup,
        "total": args.total,
        "ratio": args.ratio,
    }
    scheduler = WarmupLinearScheduler(opt, **scheduler_args)
    temp = args.temp

    train(max_epoch=args.max_epoch, eval_step=args.eval_step, save_step=args.save_step, print_step=args.print_step)
