from ultralytics import YOLO

def Task():
    # ------------------------------------------------------------------------------------------------------------------
    task = 'train'
    run_model = r'runs/detect/train/weights/last.pt'
    # ------------------------------------------------------------------------------------------------------------------
    if task == 'train':
        model = YOLO('ultralytics/cfg/models/11/OWDNet.yaml').load('yolo11n.pt')
        model.train(
            device='0',
            imgsz=640,
            data='ultralytics/cfg/datasets/OWDNet.yaml',
            epochs=300,
            batch=16,
            cache=True,
            workers=2,
            resume=False,
        )
    elif task == 'val':
        model = YOLO(run_model)
        model.val(
            imgsz=640,
            data='ultralytics/cfg/datasets/OWDNet.yaml',
            conf=0.001,
            iou=0.5,
            batch=1,
            task='test',
        )

if __name__ == '__main__':
    Task()