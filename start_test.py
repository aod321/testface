import torch
import os
from dataset import TestStage1, TwoStepData
from torchvision import transforms
from model import FirstStageModel, TwoStageModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name_list = ['eyebrows', 'eyes', 'nose', 'mouth']
# root_dir = '/home/yinzi/Downloads/datas'
txt_file = 'testing.txt'
# root_dir_2 = "/data1/yinzi/facial_parts"
root_dir = "/home/yinzi/data1/datas"
root_dir_2 = "/home/yinzi/data1/facial_parts"
state_file_root = "checkpoints_7ad8cd3d"
state_file_1 = os.path.join(state_file_root,
                            "best.pth.tar")
state_file_2 = {x: os.path.join(state_file_root,
                            "{}.pth.tar".format(x))
                for x in model_name_list
                }

# teststage1 = TestStage1(device=device, model_class=FirstStageModel,
#                         statefile=state_file_1, dataset_class=HelenDataset,
#                         txt_file=txt_file, root_dir=root_dir,
#                         batch_size=16)
#
# teststage1.start_test()

root_dir = {"image":root_dir,
           "parts": root_dir_2}
state_file_root = "checkpoints_7ad8cd3d"
state_file_1 = os.path.join(state_file_root,
                            "best.pth.tar")
state_file_2 = {x: os.path.join(state_file_root,
                            "{}.pth.tar".format(x))
                for x in model_name_list
                }

teststage1 = TestStage1(device=device, model_class=TwoStageModel,
                        statefile=state_file_1, dataset_class=TwoStepData,
                        txt_file=txt_file, root_dir=root_dir,
                        batch_size=16)

teststage1.start_test()