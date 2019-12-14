from template import TemplateModel
from model import TwoStageModel, calc_centroid
from preprocess import Resize, ToTensor, Normalize
from dataset import TwoStepData
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import uuid
import numpy as np
uuid_8 = str(uuid.uuid1())[0:8]
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=0, type=int, help="Which GPU to train.")
parser.add_argument("--batch_size", default=16, type=int, help="Batch size to use during training.")
parser.add_argument("--display_freq", default=10, type=int, help="Display frequency")
parser.add_argument("--lr", default=0.02, type=float, help="Learning rate for optimizer")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train")
parser.add_argument("--eval_per_epoch", default=1, type=int, help="eval_per_epoch ")
args = parser.parse_args()
print(args)

# Dataset Read_in Part
img_root_dir = "/data1/yinzi/datas"
# root_dir = '/home/yinzi/Downloads/datas'
part_root_dir = "/data1/yinzi/facial_parts"
root_dir = {
    'image': img_root_dir,
    'parts': part_root_dir
}
txt_file_names = {
    'train': "exemplars.txt",
    'val': "tuning.txt"
}

twostage_Dataset = {x: TwoStepData(txt_file=txt_file_names[x],
                                   root_dir=root_dir,
                                   transform=transforms.Compose([
                                       Resize((64, 64)),
                                       ToTensor(),
                                       Normalize()
                                      ])
                                   )
                    for x in ['train', 'val']
                    }

two_dataloader = {x: DataLoader(twostage_Dataset[x], batch_size=args.batch_size,
                                shuffle=True, num_workers=16)
                  for x in ['train', 'val']
                  }


class TrainModel(TemplateModel):

    def __init__(self):
        super(TrainModel, self).__init__()
        # ============== not neccessary ===============
        self.train_logger = None
        self.eval_logger = None
        self.args = args

        # ============== neccessary ===============
        self.writer = SummaryWriter('log')
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")

        # self.model = FirstStageModel().to(self.device)
        self.model = TwoStageModel().to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), self.args.lr,  momentum=0.9, weight_decay=0.0001)
        # self.optimizer = optim.Adam(self.model.parameters(), self.args.lr)
        # self.regress_criterion = nn.MSELoss()
        self.regress_criterion = nn.L1Loss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.metric = nn.BCEWithLogitsLoss()
        # self.criterion = nn.CrossEntropyLoss()
        # self.metric = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=75, gamma=0.1)

        self.train_loader = two_dataloader['train']
        self.eval_loader = two_dataloader['val']

        self.ckpt_dir = "checkpoints_%s" % uuid_8
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()


    def train_loss(self, batch):
        input, ground_truth = batch['image'].to(self.device), batch['labels'].to(self.device)
        orig = batch['orig'].to(self.device)
        parts = batch['parts'].to(self.device)
        # t_theta = GetTheta(ground_truth)
        # t_theta Shape (N, 9, 2, 3)
        # t_theta = t_theta[:, 1:9]  # Shape (N, 8, 2, 3)
        t_cen = calc_centroid(ground_truth[:, 1:9])
        t_cen = t_cen / 512.0  # remap to 0-1 range

        pred, cen = self.model(input, orig)

        loss_mask = self.criterion(pred, parts)
        loss_regress = self.regress_criterion(cen, t_cen)

        loss = loss_mask + loss_regress
        # loss = loss_regress
        # del s_theta, t_theta
        return loss, None

    def eval_error(self):
        loss_list = []
        for batch in self.eval_loader:
            input, ground_truth = batch['image'].to(self.device), batch['labels'].to(self.device)
            orig = batch['orig'].to(self.device)
            parts = batch['parts'].to(self.device)
            # t_theta = GetTheta(ground_truth)
            # t_theta Shape (N, 9, 2, 3)
            # t_theta = t_theta[:, 1:9]  # Shape (N, 8, 2, 3)
            # cen Shape (N, 8, 4)
            # s_theta = torch.zeros(t_theta.shape).to(self.device)
            # Shape(N, 8, 2, 3)
            # s_theta[:, :, 0, 0] = cen[:, :, 0]
            # s_theta[:, :, 0, 2] = cen[:, :, 2]
            # s_theta[:, :, 1, 1] = cen[:, :, 1]
            # s_theta[:, :, 1, 2] = cen[:, :, 3]

            t_cen = calc_centroid(ground_truth[:, 1:9])
            t_cen /= 512.0

            pred, cen = self.model(input, orig)

            error_mask = self.metric(pred, parts)
            # error_regress = self.regress_criterion(s_theta, t_theta)
            error_regress = self.regress_criterion(cen, t_cen)
            error = error_mask + error_regress
            # error = error_regress
            loss_list.append(error.item())

        # del s_theta, t_theta
        return np.mean(loss_list), None


def start_train():
    train = TrainModel()

    for epoch in range(args.epochs):
        train.train()
        train.scheduler.step()
        if (epoch + 1) % args.eval_per_epoch == 0:
            train.eval()

    print('Done!!!')


start_train()
