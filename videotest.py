import cv2
from ultralytics import YOLO

# Khởi tạo mô hình YOLOv8 bằng trọng số được lưu trữ trong tệp 'best.pt'
model = YOLO('runs/detect/train/weights/best.pt')

# Mở luồng video từ webcam hoặc từ một tệp video, tùy thuộc vào đường dẫn được chọn
#vid = cv2.VideoCapture(1)
vid = cv2.VideoCapture('red.mp4')

while True:
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)

    # Sử dụng mô hình YOLOv8 để nhận diện đối tượng trên khung hình
    results = model(frame)

    # Độ tin cậy tối thiểu để hiển thị đối tượng
    min_confidence = 0.8

    # Vẽ các đối tượng đã được nhận diện trên khung hình với độ tin cậy tối thiểu
    annotated_image = results[0].plot(conf=min_confidence)

    cv2.imshow("YOLOv8 Inference", annotated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
