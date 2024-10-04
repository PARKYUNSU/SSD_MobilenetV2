from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
from itertools import product as product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InvertedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t, stride=1):  # t = expansion
        super().__init__()

        self.stride = stride
        self.identity = (self.stride == 1 and in_channels == out_channels)

        # narrow -> wide
        if t != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * t, 1, bias=False),
                nn.BatchNorm2d(in_channels * t),
                nn.ReLU6(inplace=True)
            )
        # t = 1
        else:
            self.expand = nn.Identity()

        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t, bias=False),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True)
        )

        # Linear Bottleneck / wide -> narrow
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.pointwise(x)

        # Residual
        if self.identity:
            x = x + identity
        return x

class MobileNetV2Base(nn.Module):
    def __init__(self):
        super(MobileNetV2Base, self).__init__()

        # First convolution layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # Bottleneck layers with gradual channel increase
        self.bottleneck1_6 = nn.Sequential(
            self._make_stage(32, 24, t=1, n=1),   # Bottleneck 1: 32 -> 24
            self._make_stage(24, 32, t=6, n=2, stride=2),  # Bottleneck 2: 24 -> 32
            self._make_stage(32, 64, t=6, n=3, stride=2),  # Bottleneck 3: 32 -> 64
            self._make_stage(64, 128, t=6, n=4, stride=2),  # Bottleneck 4: 64 -> 128
            self._make_stage(128, 256, t=6, n=3),  # Bottleneck 5: 128 -> 256
            self._make_stage(256, 512, t=6, n=3, stride=1)  # Bottleneck 6: 256 -> 512
        )
        
        self.bottleneck7_rest = nn.Sequential(
            self._make_stage(512, 1024, t=6, n=1)  # Bottleneck 7: 512 -> 1024
        )

    def forward(self, image):
        out = self.first_conv(image) # ([1, 32, 150, 150])
        out = self.bottleneck1_6(out) #([1, 512, 19, 19])
        conv4_3_feats = out # ([1, 512, 19, 19])
        
        out = self.bottleneck7_rest(out) # Rest of Bottleneck ([1, 1024, 19, 19])
        conv7_feats = out # [1, 1024, 19, 19])
        
        return conv4_3_feats, conv7_feats
    
    def _make_stage(self, in_channels, out_channels, t, n, stride=1):
        layers = [InvertedResBlock(in_channels, out_channels, t, stride)]
        in_channels = out_channels
        for _ in range(n - 1):
            layers.append(InvertedResBlock(in_channels, out_channels, t))

        return nn.Sequential(*layers)

class AuxiliaryConvolutions(nn.Module):
    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        # Auxiliary Layers
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0) # 1x1 컨볼루션 채널 수 (1024 -> 256)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 3x3 컨볼루션, stride=2 해상도 절반, 차원 증가 (256 -> 512)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0) # 1x1 컨볼루션 채널 수 (512 -> 128)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 3x3 컨볼루션, stride=2 해상도 절반, 차원 증가 (128 -> 256)


        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)  # 1x1 컨볼루션 채널 수 (256 -> 128)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0) # 3x3 컨볼루션, stride=2 해상도 절반, 차원 증가 (128 -> 256)


        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)  # 1x1 컨볼루션 채널 수 (256 -> 128)
        self.conv11_2 = nn.Conv2d(128, 128, kernel_size=3, padding=0) # 3x3 컨볼루션, 해상도 유지 차원은 동일 (128 -> 128)

        self.init_conv2d()
    
    # 가중치 초기화
    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        out = F.relu(self.conv8_1(conv7_feats))  # [1, 256, 19, 19] 
        out = F.relu(self.conv8_2(out))
        conv8_2_feats = out  # [1, 512, 10, 10]

        out = F.relu(self.conv9_1(out))  # [1, 128, 10, 10]
        out = F.relu(self.conv9_2(out))
        conv9_2_feats = out  # [1, 256, 5, 5]

        out = F.relu(self.conv10_1(out))  # [1, 128, 5, 5]
        out = F.relu(self.conv10_2(out))
        conv10_2_feats = out  # [1, 256, 3, 3]

        out = F.relu(self.conv11_1(out))  # [1, 128, 3, 3]
        conv11_2_feats = F.relu(self.conv11_2(out))  # [1, 128, 1, 1]

        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats


class PredictionConvolutions(nn.Module):
    def __init__(self, n_classes):
        super(PredictionConvolutions, self).__init__()
        self.n_classes = n_classes
        
        # 각 피처 맵에서 사용할 prior-boxes의 개수 지정
        # feature map p = c + 4 Convolution layer
        n_boxes = {'conv4_3': 4, 'conv7': 6, 'conv8_2': 6, 'conv9_2': 6, 'conv10_2': 4, 'conv11_2': 4}
        
        # 4 = cx(centre x), cy(centtre y), w , h
        # Localization Prediction Convolutions : 각 피처 맵에서 박스 좌표 예측
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(128, n_boxes['conv11_2'] * 4, kernel_size=3, padding=1)

        # c = n_classes
        # Class Prediction Convolutions : 각 피처 맵에서 클래스 확률 예측
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(128, n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)

        # 가중치 초기화
        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        batch_size = conv4_3_feats.size(0)

        # Predict localization boxes' coordinates
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4) # [1, 1444, 4]
        # (B, 512, 38, 38) -> permute -> (B, 38, 38, 512) -> view(B, -1, 4) -> (B, 38*38*4, 4) -> coordinates of 4 pcs prior box
        
        l_conv7 = self.loc_conv7(conv7_feats)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4) # [1, 2166, 4]
        
        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4) # [1, 600, 4]
        
        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4) # [1, 150, 4]
        
        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4) # [1, 36, 4]
        
        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4) # [1, 4, 4]

        # Predict class scores
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes) # [1, 1444, 21]
        # (B, 512, 38, 38) -> (B, 38, 38, 512) -> view -> (B, 38*38*4, n_classes) -> predict class probabilities for prior boxes
        
        c_conv7 = self.cl_conv7(conv7_feats)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes) # [1, 2166, 21]
        
        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes) # [1, 600, 21]
        
        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes) # [1, 150, 21]
        
        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes) # [1, 36, 21]
        
        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes) # [1, 4, 21]

        # Concatenate all predictions
        # dim = 1 -> locs = [batch_size, total_number_of_prior_boxes, 4] , classes_scores = [batch_size, total_number_of_prior_boxes, n_classes]
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1) # [1, 4400, 4]
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim=1) # [1, 4400, 21]

        return locs, classes_scores


class SSD300(nn.Module):
    def __init__(self, n_classes):
        super(SSD300, self).__init__()

        self.n_classes = n_classes
        self.base = MobileNetV2Base()  # MobileNetV2 백본
        self.aux_convs = AuxiliaryConvolutions()  # SSD용 추가 convolution layers
        self.pred_convs = PredictionConvolutions(n_classes)  # 예측 convolution layers

         # conv4_3 L2 정규화 (Vgg feature map 고해상도이기에 스케일 조정)
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)
        # con4_3(큰해상도 19*19)가 작은 물체를 더 잘 감지하기 위해서 rescale_factors로 20배로 스케일링

        # Prior boxes 생성
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        # MobileNetV2의 feature map 추출
        conv4_3_feats, conv7_feats = self.base(image)
        
        # L2 정규화 및 Rescaling
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()
        conv4_3_feats = conv4_3_feats / norm
        conv4_3_feats = conv4_3_feats * self.rescale_factors

        # SSD 추가 레이어 적용
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)

        # Localization 및 Class Prediction
        locs, class_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats)

        return locs, class_scores

    def create_prior_boxes(self):
        # MobileNetV2에 맞는 prior box 설정 (Feature map에 맞는)
        fmap_dims = {'conv4_3': 19, 'conv7': 19, 'conv8_2': 10, 'conv9_2': 5, 'conv10_2': 3, 'conv11_2': 1}
        obj_scales = {'conv4_3': 0.1, 'conv7': 0.2, 'conv8_2': 0.375, 'conv9_2': 0.55, 'conv10_2': 0.725, 'conv11_2': 0.9}
        aspect_ratios = {
            # 1 = 정사각형, 2 = 가로가 세로보다 2배긴 직사각형 0.5 = 세로가 2배긴 직사각형
            'conv4_3': [1., 2., 0.5],
            'conv7': [1., 2., 3., 0.5, .333],
            'conv8_2': [1., 2., 3., 0.5, .333],
            'conv9_2': [1., 2., 3., 0.5, .333],
            'conv10_2': [1., 2., 0.5],
            'conv11_2': [1., 2., 0.5]
        }

        prior_boxes = []
        # 각 feature map에서 prior box 생성
        for k, fmap in enumerate(fmap_dims.keys()):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[list(fmap_dims.keys())[k + 1]])
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)
        prior_boxes.clamp_(0, 1)
        return prior_boxes



    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Eval.py
        det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                            min_score=0.01, max_overlap=0.45,
                                                                                            top_k=200)
                                                                                            
        predicted_locs : SSD가 예측한 바운딩 박스 좌표 (M, 4400, 4)
        predicted_scores : 각 prior box에 대한 클래스 확률 (N, 4400, n_classes)
        min_score : 클래스 예측 확률의 기준 값 (값 이상일 때만 객체 인식)
        max_overlap : NMS를 적용하는 겹침의 기준 값 (값 이하일 경우 바운딩 박스 제거)
        top_k : 각 이미지에서 최종적으로 남길 객체의 상위 k개만 유지
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 4400, n_classes)
        
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)
        
        # 각 배치별로 이미지 처리
        # 좌표 생성 함수 (utils.py)
        for i in range(batch_size):
            # predicted_locs : prior box를 기준으로 모델이 예측한 위치 정보
            # gcxgcy_to_cxcy : predicted_locs를 prior box 기준으로 절대 좌표 변환
            # cxcy_to_xy : (xmin, ymin, xmax, ymax) 로 변환
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (4400, 4)

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (4400)

            # 클래스별로 탐지된 객체 확인
            # 클래스 1 : 배경
            for c in range(1, self.n_classes):
                class_scores = predicted_scores[i][:, c]  # (4400) # 현재 클래스 c에 대한 점수
                score_above_min_score = class_scores > min_score  # min_score 이상 box
                n_above_min_score = score_above_min_score.sum().item() #True:1, False:0로 min_score 넘는 box 개수 sum()으로 계산
                if n_above_min_score == 0: # mins_score 넘는 box가 없으면
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 4400
                class_decoded_locs = decoded_locs[score_above_min_score]  # min_score를 넘는 상자

                
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                for box in range(class_decoded_locs.size(0)):
                    if suppress[box] == 1:
                        continue
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    suppress[box] = 0

                # 기준을 통과한 바운딩 박스, 클레스 라벨 및 클래스 확률 저장
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # 객체 탐지 안될 경우 -> 배경으로 간주되어 바운딩 박스, 클레스 라벨 저장
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # 탐지된 상자 중 상위 k개만 남기기
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores


class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        """
        priors_cxcy : Prior Boxes
        threshold : IoU 임계값 (값 이상일 때 Positive로 간주)
        neg_pos_ratio : Negative-to-positive 비율로, 논문에서는 3:1 비율
        alpha : Localizatioin Loss와 Confidence Loss 간의 가중치 설정하는 파라미터
        smooth_l1 : Localization Loss 계산을 위한 Smooth L1 Loss
        cross_entropy : Confidence Loss 계산을 위한 Cross Entropy Loss
        """
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
	
    # 모델이 예측한 Bounding Box 좌표, Class 확률
    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)
		
        # True locs, True classs 텐서 초기화후 실제 정답값들 저장
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 4400, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 4400)

        for i in range(batch_size):
            n_objects = boxes[i].size(0)
			
            # 각 이미지에 대해 prior box와 정답(Ground Truth)과 매칭
            # find_jaccard_overlap 함수로 IoU 계산
            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 4400)

            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (4400) # overlap.map(dim=0) prior box에 대해 IoU가 가장큰 객체를 선택
            
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o) # 각 정답 객체와 가장 많이 겹치는 Prior box 선택

            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            overlap_for_each_prior[prior_for_each_object] = 1. # 정답과 prior box 간의 IoU 값을 1로 설정 -> 완벽하게 맞는 것으로 간주

			# Positive or Negative 식별
            # IoU 값이 Threshold 기준 미만이면 Negarive -> 배경으로 간주
            label_for_each_prior = labels[i][object_for_each_prior]  # (4400)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (4400)

            true_classes[i] = label_for_each_prior

            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (4400, 4)

        positive_priors = true_classes != 0  # (N, 4400) # Posive Prior Box

        # LOCALIZATION LOSS
        # Positive prior box 좌표와 정답 좌표간의 차이를 Smooth L1 Loss로 계산
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # CONFIDENCE LOSS
        # Positive와 Negative에 대한 계산을 Hard Negative Mining을 사용하여 Negative 예측이 너무 많이 학습되지 않게 제어
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 4400)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 4400)

        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        conf_loss_neg = conf_loss_all.clone()  # (N, 4400)
        conf_loss_neg[positive_priors] = 0.  # (N, 4400)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 4400)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 4400)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 4400)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss
    
# 디버깅
dummy_image = torch.randn(1, 3, 300, 300).to(device)
n_classes = 21

ssd = SSD300(n_classes=n_classes).to(device)

with torch.no_grad():
    locs, class_scores = ssd(dummy_image)
print(f"Localization output shape: {locs.shape}")  # (batch_size, num_prior_boxes, 4)
print(f"Class scores output shape: {class_scores.shape}")  # (batch_size, num_prior_boxes, num_classes)