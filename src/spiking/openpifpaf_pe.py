from typing import Any

# import openpifpaf
# import torch
# import mmcv
import cv2 as cv
import os

pathdir_wsl = "/mnt/c/Users/welleron/Desktop/mmp/datasets/womens_sp/videos/"
outpath_wsl = "/mnt/c/Users/welleron/Desktop/mmp/tutorials/output/"


# def posepifpaf_pe() -> None:
#     print('OpenPifPaf version', openpifpaf.__version__)
#     print('PyTorch version', torch.__version__)
#
#     predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')
#     cap = cv.VideoCapture(os.path.join(pathdir_wsl, "1.mp4"))
#
#     width: int = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
#     height: int = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
#     fcount: int = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
#     fps: int = int(cap.get(cv.CAP_PROP_FPS))
#
#     writer = cv.VideoWriter(os.path.join(outpath_wsl, "pifpaf_output.mp4v"),
#                             cv.VideoWriter_fourcc("P", "I", "M", "1"),
#                             fps,
#                             (width, height), isColor=False)
#
#     while cap.isOpened():
#         ret: bool
#         frame: Any
#         ret, frame = cap.read()
#         image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#
#         predictions, gt_anns, meta = predictor.numpy_image(image)
#         print(predictions)
#
#         annotation_painter = openpifpaf.show.AnnotationPainter()
#         with openpifpaf.show.Canvas.image(image) as ax:
#             writer.write(annotation_painter.annotations(ax, predictions))
#             annotation_painter.annotations(ax, predictions)
#
#         cv.imshow(f"MediaPipe Pose Estimation", image)
#
#     writer.release()
#     cap.release()
#     cv.destroyAllWindows()
