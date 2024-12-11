import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

class Trainer:
    def __init__(self, model, train_loader, valid_loader, save_dir, args):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=(lambda epoch: args.anneal_base ** epoch))

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.args = args
        self.save_dir = save_dir
        self.model_dir = os.path.join(save_dir, 'checkpoint')

        # training process recording
        self.global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.best_valid_metric = 100
        self.best_valid_epoch = 0
        self.patience = 5 if args.early_stop else args.max_epoch + 1

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        with open(os.path.join(self.save_dir, 'train_config.json'), 'w') as fout:
            json.dump(self.args.__dict__, fout, indent=2)
        self.log_file = open(os.path.join(self.save_dir, "train_log.txt"), 'a+')

    @classmethod
    def to_device(cls, data, device):
        if isinstance(data, dict):
            for key in data:
                data[key] = cls.to_device(data[key], device)
        elif isinstance(data, list) or isinstance(data, tuple):
            res = [cls.to_device(item, device) for item in data]
            data = type(data)(res)
        elif hasattr(data, 'to'):
            data = data.to(device)
        return data
    

    def train(self):

        self.model.train()
        for _ in range(self.args.max_epoch):
            for batch in tqdm(self.train_loader):
                batch = self.to_device(batch, device)
                
                loss, snll, closs = self.model(batch['X'], batch['S'], batch['L'], batch['offsets'])
                self.optimizer.zero_grad()
                loss.backward()
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()

                if self.global_step % 100 == 1:
                    print("\033[0;30;46m {} | Epoch: {}, Step: {} | Train Loss: {:.5f}, SNLL: {:.5f}, Closs: {:.5f}, PPL: {:.5f}\033[0m".format(time.strftime("%Y-%m-%d %H-%M-%S"), self.epoch, self.global_step, loss.item(), snll.item(), closs.item(), snll.exp().item()))
                    self.log_file.write("{} | Epoch: {}, Step: {} | Train Loss: {:.5f}, SNLL: {:.5f}, Closs: {:.5f}, PPL: {:.5f}\n".format(time.strftime("%Y-%m-%d %H-%M-%S"), self.epoch, self.global_step, loss.item(), snll.item(), closs.item(), snll.exp().item()))
                    self.log_file.flush()

                self.global_step += 1
            self.scheduler.step()


            metric_arr = []
            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(self.valid_loader):
                    batch = self.to_device(batch, device)
                    loss, _, _ = self.model(batch['X'], batch['S'], batch['L'], batch['offsets'])
                    metric_arr.append(loss.cpu().item())
                    self.valid_global_step += 1

            valid_metric = np.mean(metric_arr)
            if valid_metric < self.best_valid_metric:
                self.patience = 5
                # torch.save(self.model, os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt'))
                torch.save(self.model, os.path.join(self.model_dir, f'best.ckpt'))
                self.best_valid_metric = valid_metric
                self.best_valid_epoch = self.epoch
            else:
                self.patience -= 1

            print("\033[0;30;43m {} | Epoch: {} | Val Loss: {:.5f}, Best Val: {:.5f}, Best Epoch: {}\033[0m".format(time.strftime("%Y-%m-%d %H-%M-%S"), self.epoch, valid_metric.item(), self.best_valid_metric, self.best_valid_epoch))
            self.log_file.write("{} | Epoch: {} | Val Loss: {:.5f}, Best Val: {:.5f}, Best Epoch: {}\n".format(time.strftime("%Y-%m-%d %H-%M-%S"), self.epoch, valid_metric.item(), self.best_valid_metric, self.best_valid_epoch))
            self.log_file.flush()

            self.epoch += 1
            if self.patience <= 0:
                print(f'Early Stopping!')
                break