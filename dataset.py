from torch.utils.data import Dataset
import jpeg4py as jpeg
import os
import numpy as np
import torchvision
from skimage import io
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from preprocess import Resize,ToTensor
from model import calc_centroid
from PIL import Image, ImageDraw
import cv2
from visualize import imshow, show_mask
import torchvision.transforms.functional as TF


class TwoStepData(Dataset):

    def __len__(self):
        return len(self.name_list)

    def __init__(self, txt_file, root_dir, transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        torchvision.set_image_backend('accimage')
        self.name_list = np.loadtxt(os.path.join(root_dir['image'], txt_file), dtype="str", delimiter=',')
        self.root_dir = root_dir
        self.img_root_dir = root_dir['image']
        self.part_root_dir = root_dir['parts']
        self.transform = transform
        self.label_name = {
              'eyebrow1',
              'eyebrow2',
              'eye1',
              'eye2',
              'nose',
              'mouth'
        }
        self.parts_range = {
            'eyebrow1': range(2, 3),
            'eyebrow2': range(3, 4),
            'eye1': range(4, 5),
            'eye2': range(5, 6),
            'nose': range(6, 7),
            'mouth': range(7, 10)
        }

    def __getitem__(self, idx):
        img_name = self.name_list[idx, 1].strip()
        img_path = os.path.join(self.img_root_dir, 'images',
                                img_name + '.jpg')
        parts_path ={
            x: [os.path.join(self.part_root_dir, '%s' % x,
                             'labels', img_name,
                             img_name + "_lbl%.2d.png" % j
                             )
                for j in self.parts_range[x]
                ]
            for x in self.label_name
        }

        # print(parts_path)
        labels_path = [os.path.join(self.img_root_dir, 'labels',
                                    img_name,
                                    img_name + "_lbl%.2d.png") % i
                       for i in range(11)]
        image = jpeg.JPEG(img_path).decode()  # [1, H, W]

        parts = [io.imread(parts_path[x][0])
                 for x in {'eye1', 'eye2', 'eyebrow1',
                           'eyebrow2', 'nose'}
                 ]
        parts.extend([io.imread(parts_path['mouth'][i])
                      for i in range(3)])

        labels = [io.imread(labels_path[i]) for i in range(11)]  # [11, 64, 64]
        labels = np.array(labels)
        bg = 255 - labels[2:10].sum(0)
        labels = np.concatenate(([bg.clip(0, 255)], labels[2:10]), axis=0)
        labels = np.uint8(labels)
        # bg = labels[0] + labels[1] + labels[10]

        sample = {'image': image, 'labels': labels, 'parts': parts, 'orig': image}

        if self.transform:
            sample = self.transform(sample)
        return sample



class TestStage(object):
    def __init__(self, device, model_class, statefile, dataset_class,
                 txt_file, root_dir, batch_size, is_shuffle=False, num_workers=4):
        # self.F1_name_list = ['eyebrows', 'eyes', 'nose', 'u_lip', 'i_mouth', 'l_lip', 'mouth_all']

        self.dataloader = None
        self.predict = None
        self.device = device
        self.model = None
        self.model_class = model_class
        self.statefile = statefile
        self.dataset_class = dataset_class
        self.dataset = None
        self.txt_file = txt_file
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle
        self.num_workers = num_workers
        self.centorids = None

    def load_model(self):
        model = self.model_class().to(self.device)
        state = torch.load(self.statefile, map_location=self.device)
        state = state['model']
        model.load_state_dict(state)
        self.model = model

    def load_dataset(self):
        pass

    def start_test(self):
        pass

    def get_predict(self, model, image):
        pred, cen = model(image.to(self.device), self.orig)
        self.predict = torch.softmax(pred, 1)
        # self.predict = (self.predict > 0.5).float()
        self.centorids = cen
        np_pred = self.predict.detach().cpu().numpy()
        return self.predict

    def get_predict_onehot(self, model, image):
        pred, cen = model(image.to(self.device))
        predict = torch.softmax(pred, 1)
        # predict Shape(N, 2, 64, 64) or (N, 4, 80, 80)
        refer = predict.argmax(dim=1, keepdim=False)  # Shape(N, 64, 64) or Shape(N, 80, 80)
        for i in range(predict.shape[1]):
            predict[:, i] = (refer == i).float()
        self.predict = predict
        self.centorids = cen
        return predict


class TestStage1(TestStage):
    def __init__(self, device, model_class, statefile,
                 dataset_class, txt_file, root_dir, batch_size):
        super(TestStage1, self).__init__(device, model_class, statefile,
                                         dataset_class, txt_file, root_dir, batch_size)
        self.F1_name_list = ['eyebrow1', 'eyebrow2',
                             'eye1', 'eye2',
                             'nose', 'u_lip', 'i_mouth', 'l_lip']
        self.model_name_list = ['eyebrows', 'eyes', 'nose', 'mouth']

        self.TP = {x: 0.0
                   for x in self.F1_name_list}
        self.FP = {x: 0.0
                   for x in self.F1_name_list}
        self.TN = {x: 0.0
                   for x in self.F1_name_list}
        self.FN = {x: 0.0
                   for x in self.F1_name_list}
        self.recall = {x: 0.0
                       for x in self.F1_name_list}
        self.precision = {x: 0.0
                          for x in self.F1_name_list}
        self.F1_list = {x: []
                        for x in self.F1_name_list}
        self.F1 = {x: 0.0
                   for x in self.F1_name_list}
        self.load_model()
        self.load_dataset()
        self.orig = None
        self.parts = None

    def load_dataset(self):
        self.dataset = self.dataset_class(txt_file='testing.txt',
                                          root_dir=self.root_dir,
                                          transform=transforms.Compose([
                                              Resize((64, 64)),
                                              ToTensor()
                                          ])
                                          )
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                     shuffle=self.is_shuffle, num_workers=self.num_workers)

    def show_centroids(self, batch, preds):
        # Input img Shape(N, 3, H, W) labels Shape(N,C,H,W) centroids Shape(N, C, 2)
        # img, labels = batch['image'], batch['labels']
        orig = batch['orig']
        labels = batch['labels']
        # pred_arg = preds.argmax(dim=1, keepdim=False)
        # binary_list = []
        # for i in range(preds.shape[1]):
        #     binary = (pred_arg == i).float()
        #     binary_list.append(binary)
        # preds = torch.stack(binary_list, dim=1)
        # pred_centroids = calc_centroid(preds)
        # np_label = labels.detach().cpu().numpy()
        pred_centroids = self.centorids * 512
        true_centroids = calc_centroid(labels[:, 1:9])
        # print("True:")
        # print(true_centroids)
        # print("Prdicts:")
        # print(pred_centroids)
        n = orig.shape[0]
        c = pred_centroids.shape[1]
        image_list = []
        for i in range(n):
            image = TF.to_pil_image(orig[i])
            draw = ImageDraw.Draw(image)
            for j in range(c):
                y_1 = torch.floor(true_centroids[i][j][0]).int().tolist()
                x_1 = torch.floor(true_centroids[i][j][1]).int().tolist()
                draw.point((x_1, y_1), fill=(0, 255, 0))
            # for k in range(c):
            #     y_2 = torch.floor(pred_centroids[i][k][0]).int().tolist()
            #     x_2 = torch.floor(pred_centroids[i][k][1]).int().tolist()
            #     draw.point((x_2, y_2), fill=(255, 0, 0))
            image_list.append(TF.to_tensor(image))
        out = torch.stack(image_list)
        out = torchvision.utils.make_grid(out)
        imshow(out)



    def start_test(self):
        for i_batch, sample_batched in enumerate(self.dataloader):
            img = sample_batched['image'].to(self.device)
            labels = sample_batched['labels'].to(self.device)
            self.orig = sample_batched['orig'].to(self.device)
            self.parts = sample_batched['parts'].to(self.device)
            np_parts = self.parts.detach().cpu().numpy()
            self.get_predict(self.model, img)
            # self.get_predict_onehot(self.model, img)
            # self.auto_select(self.orig)
            # self.show_centroids(sample_batched, self.predict)
            show_mask(img, self.predict)
            # self.calc_f1(predict=self.predict, labels=labels)
        # self.output_f1_score()

    def calc_f1(self, predict, labels):
        part_name_list = {1: 'eyebrow1', 2: 'eyebrow2', 3: 'eye1', 4: 'eye2',
                          5: 'nose', 6: 'u_lip', 7: 'i_mouth', 8: 'l_lip'}
        pred = predict.argmax(dim=1, keepdim=False).to(self.device)
        ground = labels.argmax(dim=1, keepdim=False).to(self.device)
        for i in range(1, labels.shape[1]):
            self.TP[part_name_list[i]] += ((pred == i) * (ground == i)).sum().tolist()
            self.TN[part_name_list[i]] += ((pred != i) * (ground != i)).sum().tolist()
            self.FP[part_name_list[i]] += ((pred == i) * (ground != i)).sum().tolist()
            self.FN[part_name_list[i]] += ((pred != i) * (ground == i)).sum().tolist()
        for r in self.F1_name_list:
            self.recall[r] = self.TP[r] / (
                    self.TP[r] + self.FP[r])
            self.precision[r] = self.TP[r] / (
                    self.TP[r] + self.FN[r])
            self.F1_list[r].append((2 * self.precision[r] * self.recall[r]) /
                                   (self.precision[r] + self.recall[r]))
        return self.F1_list, self.recall, self.precision

    def output_f1_score(self):
        print("Stage1 F1_scores:")
        for r in self.F1_name_list:
            self.F1[r] = np.array(self.F1_list[r]).mean()
            print("{}:{}\t".format(r, self.F1[r]))

    def auto_select(self, orig):
        # theta = torch.zeros((n, l, 2, 3)).to(self.centorids.device)
        # self.model.second_stage.theta
        # theta[:, :, 0, 0] = self.centorids[: ,:, 0]
        # theta[:, :, 0, 2] = self.centorids[: ,:, 2]
        # theta[:, :, 1, 1] = self.centorids[: ,:, 1]
        # theta[:, :, 1, 2] = self.centorids[: ,:, 3]
        # Shape (10, 8, 2, 3)
        p_theta = self.model.second_stage.theta
        n, l, _, _ = p_theta.shape
        samples = []
        for i in range(l):
            girds = F.affine_grid(theta=p_theta[:, i], size=[n, 3, 64, 64], align_corners=True)
            samples.append(F.grid_sample(input=orig, grid=girds))
        samples = torch.stack(samples, dim=0)
        # Shape(8, N, 3, 64, 64)
        print("Selected Parts:")
        for i in range(l):
            im_show = torchvision.utils.make_grid(samples[i])
            im_show = im_show.detach().cpu()
            imshow(im_show)
