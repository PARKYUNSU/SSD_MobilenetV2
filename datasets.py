import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform


class PascalVOCDataset(Dataset):
    def __init__(self, data_folder, split, keep_difficult=False):
        """
        param data_folder : 데이터 파일이 저장된 폴더
        param split : TRAIN 또는 TEST
        param keep_difficult : 탐지가 어려운 객체들을 포함할지 여부
        """
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}
        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # 이미지 경로와 객체 정보를 불러옴
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

    def __getitem__(self, i):
        # 이미지 불러오기
        # param i: 불러올 이미지의 인덱스
        image = Image.open(self.images[i], mode='r').convert('RGB')

        # 객체 정보 불러오기 (박스, 라벨, 탐지 난이도)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])
        labels = torch.LongTensor(objects['labels'])
        difficulties = torch.ByteTensor(objects['difficulties'])

        # 탐지가 어려운 객체 제외
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # 이미지와 객체 정보에 변환 적용
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        # 이미지 총 개수
        return len(self.images)

    def collate_fn(self, batch):
        """
        배치 내에 다른 크기의 객체들을 처리하기 위한 커스텀 collate 함수.
        param batch: 데이터 배치
        """
        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        # 이미지들을 하나의 텐서로 스택
        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties # (N, 3, 300, 300)
