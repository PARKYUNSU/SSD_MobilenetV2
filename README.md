# SSD

**“Single Shot MultiBox Detector” -2016**

Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg

UNC Chapel Hill, Zoox Inc, Google Inc, University of Michigan

Link to Paper: https://arxiv.org/pdf/1512.02325

----
Table of Contents

1. Introduction
2. SSD
3. Matching strategy
4. Model Architecture
5. Training Objective
6. Default boxes and aspect ratios
7. Hard Negative Mining
8. Data Augmentation
9. Code
10. SSD_MobileNetV2_300
11. Result

---

## 1. Introduction

기존 객체 탐지 시스템(Faster R-CNN)은 bounding box 생성 후 픽셀 또는 특징들을 리샘플링하여 객체를 탐지합니다.

  Faster R-CNN

  RoI Pooling(Region of Interest Pooling)

  리샘플링하는 예시로, Faster R-CNN을 보면 잠재적으로 객체가 있을 수 있는 `관심 영역`ROI (Region of Interest)을 RPN (Region Proposal Network)로 bounding box을 예측 → `관심 영역`을 동일한 지정 크기로 변환하기 위해 해당 영역을 샘플링하는 과정 ROI Pooling 수행 → ROI Pooling으로 ROI bounding box에서 픽셀 및 특징을 다시 샘플링한 후, 분류기와 네트워크에 전달하여 최종 결과를 얻음

![](https://velog.velcdn.com/images/qkrdbstn24/post/70040e56-fffb-496d-a29c-ea90949e8fbd/image.png)


입력이미지 224X224에서 관심 영역 ROI(Region of Interest)이 100X65 [52, 106, 117, 206]이라 할때, 100X65 영역이 그대로 네트워크에 들어가기엔 너무 큰 정보량을 가지고 있어서 ROI Pooling을 통해서 7X7 특징맵으로 변환하고 해당 ROI는 반올림해서 [2, 3, 4, 6] 으로 변환되어 효율적으로 학습, 그러나 [2, 3, 4, 6] 이 좌표를 역으로 다시 돌리면 기존 [52, 106, 117, 206] 좌표와 차이가 조금 난다. (이유: 반올림)

그러나 픽셀 및 특징을 리샘플링하는 과정은 연산 비용이 크고 실시간 적용에 한계가 있습니다. 

SSD는 이러한 과정을 제거하여 더 빠른 속도로 실시간 탐지가 가능하게 했습니다. 논문에서는 Faster R-CNN 속도는 느리지만 높은 정확도, YOLO 빠르지만 정확도가 다소 떨어지는 한계를 거론하면서 SSD로 객체 탐지 수행이 두 방식의 장점을 결합한 모델이라 소개합니다.

## 2. SSD

### 1) Multiscale feature maps
SSD 모델은 YOLO v1모델과 같게 1-stage detector로 하나의 통합된 네트워크로 detection을 수행합니다. YOLO v1 같은 경우 7X7X30 크기의 feature map만 사용했습니다. 단일한 scale의 feature map을 사용할 경우, 다양한 크기의 객체를 포착하기 여렵다는 단점이 있습니다. 그래서 SSD는 다양한 scale의 feature map을 사용하여 detection을 수행한다는 점이 YOLO v1 과 차이점을 가지고 있습니다.


SSD의 다양한 feature map은 Early Network와 Auxiliary Layers으로 구성됩니다.

**Early Network**

Base Network로 부르며, VGG16 백본에서 중간 부분을 사용하여 (con4_3) 다양한 특징 맵을 생성하여 추후에 제일 작은 객체를 탐지 및 클래스 점수(카테고리 점수) 예측하는데 사용됩니다.

**Auxiliary Network**

Auxiliary Network에서는 Base Network에서 나온 후 (con7_3) 여러 단계로 3X3 컨볼루션을 진행하여 점차 크기가 작아진 feature map을 생성하여, 각 층 별로 더 큰 객체를 탐지 및 클래스 점수(카테고리 점수) 예측하는 데 사용됩니다.
![](https://velog.velcdn.com/images/qkrdbstn24/post/55a0c604-5f3c-4744-8963-e9886292fdbf/image.png)


아래 그림을 보면, base network 부분은 con4_3 layer에서 38X38X512, conv7 layer에서 19X19X1024 크기의 feature map을 추출합니다.

auxiliary network 부분은 conv8_2, conv9_2, conv10_2, conv11_2 layer에서 10X10X512, 5X5X256, 3X3X256, 1X1X256 크기의 feature map을 추출합니다.
![](https://velog.velcdn.com/images/qkrdbstn24/post/b2bbc206-9b0f-4faa-a538-49d46746dc69/image.png)

### 2) Convolutional predictors for detection / Default boxes and aspect ratios
SSD는 각 각 6개층(Base+Auxiliay Network)의 feature map에서 3X3 사이즈의 커널로 p개의 값을 추출합니다. 
**p = c + 4**
각 feature map들은 다양한 aspect ratios(직사각형의 가로세로 비율: 1, 2, 0.5, 0.333)로 k개의 default bounding box를 생성하고, c 개의 클래스 점수와 default bounding box의 좌표 4개의 값을 계산합니다.
![](https://velog.velcdn.com/images/qkrdbstn24/post/85ac1005-6f85-40f6-9993-bc08858a3449/image.png)

