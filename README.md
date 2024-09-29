

**“Single Shot MultiBox Detector” -2016**

Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg

UNC Chapel Hill, Zoox Inc, Google Inc, University of Michigan

Link to Paper: https://arxiv.org/pdf/1512.02325

----
Table of Contents

1. Introduction
2. SSD
3. Training
4. Code
---

## 1. Introduction

기존 객체 탐지 시스템(Faster R-CNN)은 bounding box 생성 후 픽셀 또는 특징들을 리샘플링하여 객체를 탐지합니다.

Faster R-CNN

RoI Pooling(Region of Interest Pooling)

잠재적으로 객체가 있을 수 있는 `관심 영역`ROI (Region of Interest)을 RPN (Region Proposal Network)로 bounding box을 예측 → `관심 영역`을 동일한 지정 크기로 변환하기 위해 해당 영역을 샘플링하는 과정 ROI Pooling 수행 → ROI Pooling으로 ROI bounding box에서 픽셀 및 특징을 다시 샘플링한 후, 분류기와 네트워크에 전달하여 최종 결과를 얻음

![](https://velog.velcdn.com/images/qkrdbstn24/post/70040e56-fffb-496d-a29c-ea90949e8fbd/image.png)


입력이미지 224X224에서 관심 영역 ROI(Region of Interest)이 100X65(좌표:52, 106, 117, 206)이라 할때, 100X65 영역이 그대로 네트워크에 들어가기엔 너무 큰 정보량을 가지고 있어서 ROI Pooling을 통해서 7X7 특징맵으로 변환하고 해당 ROI는 반올림해서 [2, 3, 4, 6] 으로 변환되어 효율적으로 학습

그러나 픽셀 및 특징을 리샘플링하는 과정은 연산 비용이 크고 실시간 적용에 한계가 있습니다. 

SSD는 이러한 과정을 제거하여 더 빠른 속도로 실시간 탐지가 가능하게 했습니다. 논문에서는 Faster R-CNN 속도는 느리지만 높은 정확도, YOLO 빠르지만 정확도가 다소 떨어지는 한계를 거론하면서 SSD로 객체 탐지 수행이 두 방식의 장점을 결합한 모델이라 소개합니다.

## 2. SSD
SSD는 일정한 fixed-size collection의 Bounding Box를 NMS(Nom-maxumum Suppression)를 적용해 객체를 탐지하는 모델입니다.

SSD의 구조는 Early Network와 Auxiliary Layers으로 구성됩니다.

**Early Network**
Base Network로 부르며, VGG16 백본에서 중간 부분을 사용하여 (con4_3) 다양한 특징 맵을 생성하여 추후에 제일 작은 객체를 탐지 및 클래스 점수(카테고리 점수) 예측하는데 사용됩니다.

**Auxiliary Network**
Auxiliary Network에서는 Base Network에서 나온 후 (con7_3) 여러 단계로 3X3 컨볼루션을 진행하여 점차 크기가 작아진 feature map을 생성하여, 각 층 별로 더 큰 객체를 탐지 및 클래스 점수(카테고리 점수) 예측하는 데 사용됩니다.

### 1) Multiscale feature maps
SSD 모델은 YOLO v1 모델과 같게 1-stage detector로 하나의 통합된 네트워크로 detection을 수행합니다. YOLO v1 같은 경우 7X7X30 크기의 feature map만 사용했습니다. 단일한 scale의 feature map을 사용할 경우, 다양한 크기의 객체를 포착하기 여렵다는 단점이 있습니다. 그래서 SSD는 다양한 scale의 feature map을 사용하여 detection을 수행한다는 점이 YOLO v1 과 차이점을 가지고 있습니다.

![](https://velog.velcdn.com/images/qkrdbstn24/post/55a0c604-5f3c-4744-8963-e9886292fdbf/image.png)

SSD에서는 Auxiliary Network 부분으로 4개의 conv를 통해서 다양한 크기의 feature map을 생성합니다.

아래 그림을 보면, base network 부분은 con4_3 layer에서 38X38X512, conv7 layer에서 19X19X1024 크기의 feature map을 추출합니다.

auxiliary network 부분은 conv8_2, conv9_2, conv10_2, conv11_2 layer에서 10X10X512, 5X5X256, 3X3X256, 1X1X256 크기의 feature map을 추출합니다.
![](https://velog.velcdn.com/images/qkrdbstn24/post/b2bbc206-9b0f-4faa-a538-49d46746dc69/image.png)

### 2) Convolutional predictors for detection / Default boxes and aspect ratios
SSD는 각 각 6개층(Base+Auxiliart Network)의 feature map에서 3X3 사이즈의 커널로 p개의 값을 추출합니다. 

**p = c + 4 (Class + (중심 좌표와의 x, y offset, 영역의 w, h offset))**

각 feature map들은 다양한 aspect ratios(직사각형의 가로세로 비율: 1, 2, 0.5, 0.333)로 k개의 default bounding box를 생성하고, c 개의 클래스 점수와 default bounding box의 좌표 4개의 값을 계산합니다.

그렇게 해서 SSD는 총 8732(Bounding boxes, class scores)를 얻습니다. 
![](https://velog.velcdn.com/images/qkrdbstn24/post/85ac1005-6f85-40f6-9993-bc08858a3449/image.png)


## 3. Training

### 1) Matching Strategy
 학습 과정에서 각 Default box가 어떤 Ground truth 와 매칭하는지 결정해야 합니다. 각 Default box는 위치, aspect ratios(비율), 크기가 다르게 나오게되며, 정답지인 Ground truth와 매칭하여 Jaccard Overlab이 가장 높은 박스를 선택하게 됩니다. 그런 다음, 겹쳐진 비율이 0.5 이상인 Default box들을 Ground truth와 최종 매칭합니다.
![](https://velog.velcdn.com/images/qkrdbstn24/post/7038bd51-7e59-451b-8f2e-f7eebd2715ba/image.png)
 
 Jaccard Index 또는 Jaccard Overlap 또는 IOU(Intersection-over-Union)이라 불리며, 두 박스가 얼마나 겹치는지를 측정하는 지표로, IoU 값이 1이면 두 박스가 동일, 0 이면 두 박스가 겹쳐지지 않는 것을 의미합니다.

### 2) Traning Objective
SSD의 학습 목표는, 위에서 설명한 다양한 Box들인 MultiBox에서 유래한 손실함수로, Localization Loss(loc)와 Confidence Loss(conf)의 가중치 합으로 이루어집니다.

![](https://velog.velcdn.com/images/qkrdbstn24/post/43eda36c-7979-4d54-a69a-612dedd5f1f3/image.png)
 L(x, c, l, g): 전체 손실
 x: 각 Default box가 어떤 클래스와 매칭되는지 나타내는 값
 c: 클래스 예측 값
 l: Localization(예측된 박스 위치 정보)
 g: Ground truth(정답 박스)
 α: Localization 가중치 (L_conf 와 L_loc 비율을 조정하는 역할을 합니다)
![](https://velog.velcdn.com/images/qkrdbstn24/post/825afcf2-86ea-44ea-b148-eeab5ca8ec7d/image.png)
예측된 바운딩 박스와 실제 바운딩 박스의 좌표 간의 차이를 계산합니다. 이때 중심 좌표 **(cx, cy)**와 크기 **(w, h)**의 차이를 각각 정규화하고, 이 차이에 대해 Smooth L1 손실을 적용하여 계산합니다.

![](https://velog.velcdn.com/images/qkrdbstn24/post/3fcf5f0a-efff-4b72-96fb-39a403b609c5/image.png)
예측한 클래스 확률과 실제 레이블 간의 차이를 측정합니다. 양성 샘플은 객체의 정확한 클래스를 예측하도록 학습되고, 음성 샘플은 객체가 아닌 배경을 예측하도록 학습됩니다.

i ∈ Pos는 양성(positive) 샘플, 즉 객체와 매칭된 디폴트 박스에 대해 계산 (예측 박스 i가 실제 객체 j의 클래스 p와 매칭 되었을 때 1인 지표)

i ∈ Neg는 음성(negative) 샘플, 즉 객체와 매칭되지 않은 디폴트 박스에 대해 계산(예측 박스 i가 객체가 없는 상태(즉, 배경)일 확률을 계산, 음석객체는 객체가 아닌 배경을 예측하도록 학습

### 3) Choosing scales and aspect ratios for default boxes

SSD 모델에서 **Default Boxes** 의 **Aspect ratio** 로 각 feature map에서 다양한 크기와 비율로 설정된 default box를 통해, 모델은 여러 크기의 객체를 적응하여 탐지할 수 있습니다.

SSD는 각 feature map의 크기에 맞춰 작은 객체부터 큰 객체까지 모두 탐지합니다. 첫 번째 feature map에 대해 가장 작은 스케일을, 마지막 feature map에 대해 가장 큰 스케일을 적용하빈다.

예를 들어, 첫 번째 feature map(conv4_3)의 default box는 상대적으로 작은 스케일을 가지며, 마지막 feature map(conv11_2)의 default box는 큰 스케일을 가집니다.

$s_k = s_{\text{min}} + \frac{(s_{\text{max}} - s_{\text{min}})}{(m - 1)} \cdot (k - 1)$.

여기서,

s_min과 s_max는 최소 및 최대 스케일 값,
m은 feature map의 개수,
k는 현재 feature map의 index를 나타냅니다.
Aspect Ratios

SSD는 aspect ratio로 1:1, 2:1, 3:1, 1:2, 1:3. 이를 통해 정사각형과 직사각형 등의 다양한 형태의 객체를 탐지할 수 있습니다.

1:1 aspect ratio는 정사각형 모양의 박스를 생성하고, 2:1 또는 1:2와 같은 비율은 직사각형 모양의 박스를 생성합니다.


### 4) Hard Negative Mining
Hard Negative Mining은 양성(positive) 예시와 음성(negative) 예시 간의 불균형을 해결하는 방법입니다.
 위의 Confidence Loss(conf)에서 구한 양성(객체와 일치하는 박스) 음성(객체와 매칭되지 않는 박스(배경))으로 양성대 음성 비율을 1:3 으로 조정해서 불균형을 해결 하였습니다.
 
### 5) Data Augmentation
SSD 데이터 증강으로는
- 학습 이미지 전체를 그대로 사용
- Jaccard Overlap 기준 샘플링
	기준을 0.1, 0.3, 0.5, 0.7, 0.9 중 하나로 설정됩니다. 예를 들어, IoU가 0.5인 패치를 샘플링할 경우, 추출한 패치가 객체와 50% 정도 겹치도록 만듭니다. 이 방식은 모델이 객체를 부분적으로 인식할 수 있도록 도와줍니다.
- 랜덤 샘플링
	특정 기준없이 랜덤하게 샘플링
    

### 추가적인 설명

#### 1) Aspect ratio
<img src="https://velog.velcdn.com/images/qkrdbstn24/post/dbacd609-c3e6-4cb9-8612-1fbd629c837c/image.png" width="200"/>


ex) conv4_3 피처 맵:
- object 스케일은 0.1
- aspect ratio [1., 2., 0.5]
 1 = 정사각형, 2 = 가로가 세로보다 2배긴 직사각형(빨간색 box) 0.5 = 세로가 2배긴 직사각형(파란색 box)
 
각 위치에서 prior box의 크기는 다음과 같이 계산됩니다.

1. **Aspect ratio 1**:
   - width: $0.1 \times \sqrt{1} = 0.1$
   - height: $\frac{0.1}{\sqrt{1}} = 0.1$

2. **Aspect ratio 2**:
   - width: $0.1 \times \sqrt{2} \approx 0.1414$
   - height: $\frac{0.1}{\sqrt{2}} \approx 0.0707$

3. **Aspect ratio 0.5**:
   - width: $0.1 \times \sqrt{0.5} \approx 0.0707$
   - height: $\frac{0.1}{\sqrt{0.5}} \approx 0.1414$


#### 2) Object scale
<img src="https://velog.velcdn.com/images/qkrdbstn24/post/dad76e79-5bd3-4d7e-a1d8-24b188a11990/image.png" width="200"/>

위 그림은 같은 aspect ratio를 가지진만 다른 Scale로  다른 Box 가 나오는 예시

'conv4_3': 0.1
'conv7': 0.2
'conv8_2': 0.375
'conv9_2': 0.55
'conv10_2': 0.725
'conv11_2': 0.9
<img src="https://velog.velcdn.com/images/qkrdbstn24/post/9f320e6b-235a-4810-b2e1-109e190308e3/image.png" width="400"/>
$m$ : 몇 개의 feature map (con7 ~ conv11_2  5개)
$s_k$ : scalre of $k^{th}$ layer
s<sub>min</sub> : 0.2 (단, PASCAL VOC 2007 에서 conv4_3 의 scale 을 0.1 로 setting)
s<sub>max</sub> : 0.9


1. $s_0$ = 0.1 (PASCAL VOC 2007 에서 conv4_3 의 scale 을 0.1 로 setting)

2. $s_1$ = s<sub>min</sub> = 0.2 (k=1 이므로 뒤의항 없어짐)

3. $s_2$ = $s_{\min} + \frac{(s_{\max} - s_{\min})(k - 1)}{m - 1}$ = $0.2 + \frac{(0.9 - 0.2) \cdot (2 - 1)}{5 - 1}$ = $0.2 + 0.7 \cdot \frac{1}{4}$ = 0.375

4. $s_3$ = $s_{\min} + \frac{(s_{\max} - s_{\min}) \cdot (k - 1)}{m - 1}$ = $0.2 + \frac{(0.9 - 0.2) \cdot (3 - 1)}{5 - 1}$= $0.2 + 0.7 \cdot \frac{2}{4}$ = 0.55

5. $s_4$ =$s_{\min} + \frac{(s_{\max} - s_{\min}) \cdot (k - 1)}{m - 1}$ = $0.2 + \frac{(0.9 - 0.2) \cdot (4 - 1)}{5 - 1}$ = $0.2 + 0.7 \cdot \frac{3}{4}$ = 0.725

6. $s_5$ = $s_{\min} + \frac{(s_{\max} - s_{\min}) \cdot (k - 1)}{m - 1}$ = $0.2 + \frac{(0.9 - 0.2) \cdot (5 - 1)}{5 - 1}$ = $0.2 + 0.7 \cdot \frac{4}{4}$ = 0.9

따라서 conv4_3 부터 conv11_2 까지 각 [0.1, 0.2, 0.375, 0.55, 0.725, 0.9] 의 scale 을 갖습니다. 


#### 3) Prior box
각 feature map의 크기와 각 feature map의 n_boxes의 갯수로 Total prior box 계산

n_boxes 갯수 및 feature map 크기
1. 'conv4_3': 4 (38 x 38) = 38 x 38 x 4 = 5776 -> 작은 객체 탐지
2. 'conv7': 6 (19 x 19) = 19 x 19 x 6 = 2166 -> 중간 크기 객체 탐지
3. 'conv8_2': 6 (10 x 10) = 10 x 10 x 6 = 600 -> 중간 크기 객체
4. 'conv9_2': 6 (5 x 5) = 5 x 5 x 6 = 150 -> 더 큰 객체 탐지
5. 'conv10_2': 4 (3 x 3) = 3 x 3 x 4 = 36 -> 큰 객체 탐지
6. 'conv11_2': 4 (1 x 1) = 1 x 1 x 4 = 4 -> 매우 큰 객체 탐지

Total prior box = 8732

