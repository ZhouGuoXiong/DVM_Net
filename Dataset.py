import os
import cv2
import torch.utils.data


def read_directory(directory_name, label):
    array_of_img = []
    files = os.listdir(directory_name)

    files.sort(key=lambda x: int(x[0:-4]))
    for filename in files:
        img = cv2.imread(directory_name + "/" + filename)
        if label:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)

        array_of_img.append(img)
    return array_of_img


dataset_LEVIR = 'images/LEVIR_CD'
dataset_WHU = 'images/WHU_1'
dataset_Gz = 'images/Google_Data'
dataset_SYSU_CD = 'images/SYSU-CD'
dataset_ChangeDetectionDataset = 'images/ChangeDetectionDataset'
dataset_train_1 = '/train/A'
dataset_train_2 = '/train/B'
dataset_train_label = '/train/label'
dataset_test_1 = '/test/A'
dataset_test_2 = '/test/B'
dataset_test_label = '/test/label'




class LevirWhuGzDataset(torch.utils.data.Dataset):
    def __init__(self, move='train', dataset='LEVIR_CD', transform=None):
        super(LevirWhuGzDataset, self).__init__()
        seq_img_1 = []
        seq_img_2 = []
        seq_label = []
        if dataset == 'LEVIR_CD':
            if move == 'train':

                seq_img_1 = read_directory(dataset_LEVIR + dataset_train_1, label=False)
                seq_img_2 = read_directory(dataset_LEVIR + dataset_train_2, label=False)
                seq_label = read_directory(dataset_LEVIR + dataset_train_label, label=True)
            elif move == 'test':

                seq_img_1 = read_directory(dataset_LEVIR + dataset_test_1, label=False)
                seq_img_2 = read_directory(dataset_LEVIR + dataset_test_2, label=False)
                seq_label = read_directory(dataset_LEVIR + dataset_test_label, label=True)
        elif dataset == 'WHU_1':
            if move == 'train':
                seq_img_1 = read_directory(dataset_WHU + dataset_train_1, label=False)
                seq_img_2 = read_directory(dataset_WHU + dataset_train_2, label=False)
                seq_label = read_directory(dataset_WHU + dataset_train_label, label=True)
            elif move == 'test':
                seq_img_1 = read_directory(dataset_WHU + dataset_test_1, label=False)
                seq_img_2 = read_directory(dataset_WHU + dataset_test_2, label=False)
                seq_label = read_directory(dataset_WHU + dataset_test_label, label=True)
        elif dataset == 'Google_Data':
            if move == 'train':
                seq_img_1 = read_directory(dataset_Gz + dataset_train_1, label=False)
                seq_img_2 = read_directory(dataset_Gz + dataset_train_2, label=False)
                seq_label = read_directory(dataset_Gz + dataset_train_label, label=True)
            elif move == 'test':
                seq_img_1 = read_directory(dataset_Gz + dataset_test_1, label=False)
                seq_img_2 = read_directory(dataset_Gz + dataset_test_2, label=False)
                seq_label = read_directory(dataset_Gz + dataset_test_label, label=True)
        elif dataset == 'ChangeDetectionDataset':
            if move == 'train':
                seq_img_1 = read_directory(dataset_ChangeDetectionDataset + dataset_train_1, label=False)
                seq_img_2 = read_directory(dataset_ChangeDetectionDataset + dataset_train_2, label=False)
                seq_label = read_directory(dataset_ChangeDetectionDataset + dataset_train_label, label=True)
            elif move == 'test':
                seq_img_1 = read_directory(dataset_ChangeDetectionDataset + dataset_test_1, label=False)
                seq_img_2 = read_directory(dataset_ChangeDetectionDataset + dataset_test_2, label=False)
                seq_label = read_directory(dataset_ChangeDetectionDataset + dataset_test_label, label=True)
        elif dataset == 'SYSU_CD':
            if move == 'train':
                seq_img_1 = read_directory(dataset_SYSU_CD + dataset_train_1, label=False)
                seq_img_2 = read_directory(dataset_SYSU_CD + dataset_train_2, label=False)
                seq_label = read_directory(dataset_SYSU_CD + dataset_train_label, label=True)
            elif move == 'test':
                seq_img_1 = read_directory(dataset_SYSU_CD + dataset_test_1, label=False)
                seq_img_2 = read_directory(dataset_SYSU_CD + dataset_test_2, label=False)
                seq_label = read_directory(dataset_SYSU_CD + dataset_test_label, label=True)

        self.seq_img_1 = seq_img_1
        self.seq_img_2 = seq_img_2
        self.seq_label = seq_label
        self.transform = transform

    def __getitem__(self, index):
        imgs_1 = self.seq_img_1[index]
        imgs_2 = self.seq_img_2[index]
        label = self.seq_label[index]
        if self.transform is not None:
            imgs_1 = self.transform(imgs_1)
            imgs_2 = self.transform(imgs_2)
            label = self.transform(label)
        return imgs_1, imgs_2, label

    def __len__(self):
        return len(self.seq_label)
