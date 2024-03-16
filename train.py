from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# model = YOLO('yolov8tiny2.yaml').load('yolov8n.pt')
#training
model.train(data='datas.yaml',epochs=20,imgsz=640,batch=24,optimizer='Adam')