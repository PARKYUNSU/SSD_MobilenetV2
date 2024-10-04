from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습된 모델 체크포인트 로드
checkpoint = '/Users/parkyunsu/gitfile/SSD/SSDv4_checkpoints/checkpoint_ssd300_epoch_232.pth.tar'
checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()  # 평가 모드로 전환

# 이미지 변환 정의
resize = transforms.Resize((300, 300))  # 300x300 크기로 조정
to_tensor = transforms.ToTensor()  # 텐서로 변환
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 정규화
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    param original_image : 원본 이미지
    param min_score : 객체로 인식되기 위한 최소 점수 임계값
    param max_overlap : Non-Maximum Suppression (NMS)에서 겹칠 수 있는 최대 값
    param top_k : 감지된 객체 중 상위 k개만 유지
    param suppress : 이미지에서 감지되지 않기를 원하는 클래스 목록 (옵션)
    """

    # 이미지 변환
    image = normalize(to_tensor(resize(original_image)))

    # 디바이스로 이동
    image = image.to(device)

    # 모델을 통해 예측 수행
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # SSD 출력에서 객체 감지
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # 감지된 객체를 CPU로 이동
    det_boxes = det_boxes[0].to('cpu')

    # 원본 이미지 크기로 변환
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # 클래스 라벨 디코딩
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # 객체가 감지되지 않으면 원본 이미지를 반환 -> 배경
    if det_labels == ['background']:
        return original_image

    # 이미지에 주석 추가
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    # 감지된 각 객체에 대해 상자 및 레이블을 그리기
    for i in range(det_boxes.size(0)):
        # suppress 파라미터가 설정된 경우 해당 클래스는 건너뜀
        if suppress is not None and det_labels[i] in suppress:
            continue

        # Box
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[i]])
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[det_labels[i]])
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[det_labels[i]])

        # Text (라벨)
        text_bbox = font.getbbox(det_labels[i].upper())
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)
    del draw

    return annotated_image


if __name__ == '__main__':
    # 입력 이미지 RGB로 변환
    img_path = '/Users/parkyunsu/Downloads/86778_35552_552.jpg'
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')

    detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()


