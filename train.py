import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
from tqdm import tqdm
import os

# 기본 설정
data_folder = '/kaggle/working/'
keep_difficult = True  # 'difficult' 객체도 학습에 포함할지 여부

n_classes = len(label_map)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 설정
checkpoint = None
batch_size = 8
iterations = 120000
workers = 4
lr = 1e-3
decay_lr_at = [80000, 100000]
decay_lr_to = 0.1
momentum = 0.9
weight_decay = 5e-4
grad_clip = None

cudnn.benchmark = True  # CUDA 성능 최적화

def main():
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    checkpoint_path = '/kaggle/input/ckckck5/checkpoint_ssd300_epoch_200.pth.tar'

    # 체크포인트가 존재하지 않으면 학습을 처음부터 시작
    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint file not found at {checkpoint_path}. Starting training from scratch.")
        checkpoint = None
    else:
        checkpoint = checkpoint_path

    if checkpoint is None:
        # 모델 초기화
        start_epoch = 0
        model = SSD300(n_classes=n_classes)

        # 옵티마이저 설정
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print(f'\nLoaded checkpoint from epoch {start_epoch}.\n')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 모델과 손실함수를 디바이스로 이동
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # 데이터셋 로드
    train_dataset = PascalVOCDataset(data_folder, split='train', keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)

    # 총 학습 에포크 계산
    final_epoch = 232
    epochs = iterations // (len(train_dataset) // 32)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    # 학습 진행
    for epoch in range(start_epoch, epochs + 1):
        # 학습률 감소
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # 에포크 학습
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              epochs=epochs)

        # 체크포인트 저장
        if epoch % 50 == 0 or epoch == final_epoch:
            save_checkpoint(epoch, model, optimizer, '/kaggle/working/')

def train(train_loader, model, criterion, optimizer, epoch, epochs):
    model.train()  # 학습 모드 설정
    losses = AverageMeter()  # 손실 기록

    # tqdm progress bar 설정
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch', leave=True, dynamic_ncols=True) as pbar:
        for i, (images, boxes, labels, _) in enumerate(train_loader):
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # 모델 순전파
            predicted_locs, predicted_scores = model(images)

            # 손실 계산
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)

            # 역전파 및 파라미터 업데이트
            optimizer.zero_grad()
            loss.backward()

            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)

            optimizer.step()

            # 손실 기록 업데이트
            losses.update(loss.item(), images.size(0))

            # tqdm progress bar 업데이트
            pbar.set_postfix({
                'Loss': f'{losses.val:.4f} ({losses.avg:.4f})'
            })

            pbar.update(1)

        del predicted_locs, predicted_scores, images, boxes, labels

if __name__ == '__main__':
    main()
