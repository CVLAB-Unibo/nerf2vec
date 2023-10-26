from collections import defaultdict
import os
import copy
import time
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Any, Dict, List, Tuple
from classification import config_classifier as config

import numpy as np

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import models
from torch import Tensor
import torch

import imageio.v2 as imageio
import torch.nn.functional as F

from models.baseline_classifier import Resnet50Classifier
import wandb
import datetime
from torchmetrics.classification.accuracy import Accuracy

class BaselineDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        all_files = os.listdir(self.root)
        self.item_paths = sorted([file for file in all_files if file.lower().endswith(".png")])
        # self.item_paths = sorted(self.root.glob("*.png"), key=lambda x: str(x.stem))

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image_name = self.item_paths[index]
        image_path = os.path.join(self.root, image_name)
        rgba_np = imageio.imread(image_path)
        rgba = torch.from_numpy(rgba_np).permute(2, 0, 1).float() / 255.0  # (channels, width, height)
        
        class_label = image_name.split('_')[0]
        class_id = np.array(config.LABELS_TO_IDS[class_label])
        class_id = torch.from_numpy(class_id).long()
        
        return rgba, class_id, self.item_paths[index]

class BaselineClassifier:
    def __init__(self, device='cuda:0') -> None:

        dset_root = Path(config.BASELINE_DIR)
        train_dset = BaselineDataset(dset_root, config.TRAIN_SPLIT)

        train_bs = config.TRAIN_BS
        self.train_loader = DataLoader(train_dset, batch_size=train_bs, num_workers=8, shuffle=True)

        val_bs = config.VAL_BS
        val_dset = BaselineDataset(dset_root, config.VAL_SPLIT)
        self.val_loader = DataLoader(val_dset, batch_size=val_bs, num_workers=8)

        test_dset = BaselineDataset(dset_root, config.TEST_SPLIT)
        self.test_loader = DataLoader(test_dset, batch_size=val_bs, num_workers=8)

        self.num_classes = config.NUM_CLASSES

        net =  Resnet50Classifier(config.NUM_CLASSES, interpolate_features=False)
        self.net = net.to(device)

        self.optimizer = AdamW(self.net.parameters(), config.LR, weight_decay=config.WD)
        num_steps = config.NUM_EPOCHS * len(self.train_loader)
        self.scheduler = OneCycleLR(self.optimizer, config.LR, total_steps=num_steps)

        self.epoch = 0
        self.global_step = 0
        self.best_acc = 0.0

        self.device = device

        self.ckpts_path = Path(config.BASELINE_OUTPUT_DIR) / "ckpts"

        # TODO: RESTORE THIS!
        if self.ckpts_path.exists():
            self.restore_from_last_ckpt()
        
        # TODO: REMOVE THIS!!!!
        self.config_wandb()
        predictions, true_labels = self.val("test", best=False)
        exit()

        self.ckpts_path.mkdir(exist_ok=True)
    
    def logfn(self, values: Dict[str, Any]) -> None:
        wandb.log(values, step=self.global_step, commit=False)

    def train(self) -> None:

        self.config_wandb()

        num_epochs = config.NUM_EPOCHS
        start_epoch = self.epoch

        for epoch in range(start_epoch, num_epochs):
            start = time.time()
            self.epoch = epoch

            self.net.train()
            for batch in self.train_loader:
                images, labels, _ = batch
                images = images.cuda()
                labels = labels.cuda()

                pred, labels = self.net(images, labels)
                loss = F.cross_entropy(pred, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if self.global_step % 10 == 0:
                    self.logfn({"train/loss": loss.item()})
                    self.logfn({"train/lr": self.scheduler.get_last_lr()[0]})

                self.global_step += 1
            
            end = time.time()
            print(f'epoch {epoch} completed in {end-start}s')
            
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                self.val("train")
                self.val("val")
                self.save_ckpt()
            
            if epoch == num_epochs - 1:
                predictions, true_labels = self.val("test", best=True)
                self.log_confusion_matrix(predictions, true_labels)
    
    @torch.no_grad()
    def val(self, split: str, best: bool = False, voting_enabled=True) -> Tuple[Tensor, Tensor]:
        acc = Accuracy("multiclass", num_classes=self.num_classes).cuda()
        # predictions = []
        # true_labels = []
        data_dirs = []

        predictions_tensor = torch.empty((0, self.num_classes), device=self.device)
        true_labels_tensor = torch.empty(0, device=self.device)

        if split == "train":
            loader = self.train_loader
        elif split == "val":
            loader = self.val_loader
        else:
            loader = self.test_loader

        if best:
            model = self.best_model
        else:
            model = self.net
        model = model.cuda()
        model.eval()

        losses = []

        for batch in loader:
            images, labels, dirs = batch
            images = images.cuda()
            labels = labels.cuda()

            pred, _ = self.net(images)
            loss = F.cross_entropy(pred, labels)
            losses.append(loss.item())

            pred_softmax = F.softmax(pred, dim=-1)
            acc(pred_softmax, labels)

            # predictions.append(pred_softmax.clone())
            # true_labels.append(labels.clone())
            
            predictions_tensor = torch.cat([predictions_tensor, pred_softmax.clone()], dim=0)
            true_labels_tensor = torch.cat([true_labels_tensor, labels.clone()], dim=0)
            data_dirs = data_dirs + list(dirs)


        # ################################################################################
        # TEMPORARY CODE: WIP
        # ################################################################################
        if voting_enabled:
            
            # Frequency counting operation
            actual_predictions = {}
            actual_true_labels = {}
            for data_dir, pred, label in zip(data_dirs, predictions_tensor, true_labels_tensor):
                nerf_id = f"{data_dir.split('_')[0]}_{data_dir.split('_')[1]}"

                if nerf_id not in actual_predictions:
                    actual_predictions[nerf_id] = [0] * self.num_classes
                    # actual_predictions[nerf_id] = torch.zeros(self.num_classes, device=pred.device)  # TODO: __D CHECK DIMENSION!!!!!
                    actual_true_labels[nerf_id] = label.item()

                # pred_softmaxed = torch.softmax(pred, dim=0)  # TODO: __D CHECK DIMENSION!!!!!
                # actual_predictions[nerf_id] += pred_softmaxed

                curr_pred = torch.argmax(pred).item()
                actual_predictions[nerf_id][curr_pred] += 1

            # Voting operation
            for nerf_id, values in actual_predictions.items():
                class_id = np.argmax(values)
                # class_id = torch.argmax(values).item()
                actual_predictions[nerf_id] = class_id

            # Accuracy computation
            correct_predictions = (np.array(list(actual_predictions.values())) == np.array(list(actual_true_labels.values()))).sum().item()
            total_predictions = len(actual_predictions)
            accuracy = torch.tensor((correct_predictions / total_predictions), dtype=torch.float32)

        # ################################################################################
        else:
            accuracy = acc.compute()

        self.logfn({f"{split}/acc": accuracy})
        self.logfn({f"{split}/loss": torch.mean(torch.tensor(losses))})

        if accuracy > self.best_acc and split == "val":
            self.best_acc = accuracy
            self.save_ckpt(best=True)
            self.best_model = copy.deepcopy(self.net)

        # return torch.cat(predictions, dim=0), torch.cat(true_labels, dim=0)
    
    def log_confusion_matrix(self, predictions: Tensor, labels: Tensor) -> None:
        conf_matrix = wandb.plot.confusion_matrix(
            probs=predictions.cpu().numpy(),
            y_true=labels.cpu().numpy(),
            class_names=[str(i) for i in range(config.NUM_CLASSES)],
        )
        self.logfn({"conf_matrix": conf_matrix})

    def save_ckpt(self, best: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "best_acc": self.best_acc,
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        for previous_ckpt_path in self.ckpts_path.glob("*.pt"):
            if "best" not in previous_ckpt_path.name:
                previous_ckpt_path.unlink()

        ckpt_path = self.ckpts_path / f"{self.epoch}.pt"
        torch.save(ckpt, ckpt_path)

        if best:
            ckpt_path = self.ckpts_path / "best.pt"
            torch.save(ckpt, ckpt_path)

    def restore_from_last_ckpt(self) -> None:
        if self.ckpts_path.exists():
            ckpt_paths = [p for p in self.ckpts_path.glob("*.pt") if "best" not in p.name]
            error_msg = "Expected only one ckpt apart from best, found none or too many."
            assert len(ckpt_paths) == 1, error_msg

            ckpt_path = ckpt_paths[0]
            ckpt = torch.load(ckpt_path)

            self.epoch = ckpt["epoch"] + 1
            self.global_step = self.epoch * len(self.train_loader)
            self.best_acc = ckpt["best_acc"]

            self.net.load_state_dict(ckpt["net"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
    
    def config_wandb(self):
        wandb.init(
            entity='dsr-lab',
            project='nerf2vec_classifier',
            name=f'run_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
            config=config.WANDB_CONFIG
        )