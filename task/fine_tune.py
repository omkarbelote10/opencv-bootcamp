# from ultralytics import YOLO

# def main():

#     #Train model
#     model = YOLO("yolo11m.pt")
#     model.train(
#         data="VisDrone.yaml",
#         epochs=100,
#         imgsz=640,
#         batch=4,
#         device=0,
#         workers=2,
#         optimizer="AdamW",
#         lr0=0.001,
#         cos_lr=True,
#         close_mosaic=10,
#         cache=False,
#         amp=True
#     )

# if __name__ == "__main__":
#     main()