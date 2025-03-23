import os
import time
import torch
import pyexr
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import pyrealsense2 as rs
from torchvision.ops import box_convert
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from realsense_camera import RealSenseCamera
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
# TEXT_PROMPT = ["knife","kettle"]
TEXT_PROMPT = "knife, kettle"
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor
import time

# build grounding dino model from local path
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device="cuda"
)
sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# cap = cv2.VideoCapture(0)
camera = RealSenseCamera()

if_init = False

# prompt grounding dino to get the box coordinates on specific frame

rgb_image = camera.capture_image("rgb")
# 保存图片
cv2.imwrite("rgb.jpg", rgb_image)
_, image_tensor = load_image("rgb.jpg")
boxes, confidences, labels = predict(
    model=grounding_model,
    image=image_tensor,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
)

# process the box prompt for SAM 2
h, w, _ = rgb_image.shape
boxes = boxes * torch.Tensor([w, h, w, h])
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
confidences = confidences.numpy().tolist()
class_names = labels
print(labels)
print(input_boxes)
boxes = input_boxes

count=0

from multiprocessing import Process, Manager, Lock
import numpy as np
import time

def writer(shared_data, lock):
    for i in range(5):  # 模拟多次写入
        with lock:  # 加锁
            shared_data["depth"] = np.random.rand(480, 640).astype(np.float32)
            shared_data["color"] = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            shared_data["mask"] = np.random.randint(0, 2, (480, 640), dtype=np.uint8)
            print(f"Writer: Updated data at iteration {i}")
        time.sleep(1)  # 模拟写入延迟

if __name__ == "__main__":
    with Manager() as manager:
        shared_data = manager.dict()
        lock = Lock()
        p1 = Process(target=writer, args=(shared_data, lock))
        p1.start()
        p1.join()

while True:
    # frame_interval = 1.0 / 20  # 5Hz
    # time.sleep(frame_interval)
    # ret, frame = cap.read()
    # if not ret:
    #     break
    rgb_image = camera.capture_image("rgb")
    depth_image = camera.capture_image("depth")
    frame =rgb_image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    width, height = frame.shape[:2][::-1]
    if not if_init:
        predictor.load_first_frame(frame)
        if_init = True
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        # Let's add a positive click at (x, y) = (210, 350) to get started


        ##! add points, `1` means positive click and `0` means negative click
        # points = np.array([[660, 267]], dtype=np.float32)
        # labels = np.array([1], dtype=np.int32)

        # _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
        # )


        ## ! add bbox
        # box=[[600, 214], [765, 286]]
        
        for object_id, box in enumerate(boxes):
            print("object_id",object_id)
            bbox = np.array(box, dtype=np.float32)
            _, out_obj_ids, out_mask_logits= predictor.add_new_prompt(
                frame_idx=ann_frame_idx, 
                obj_id = object_id, #ann_obj_id, 
                bbox=bbox
            )

        print("out_obj_ids",out_obj_ids)
        ##! add mask
        # mask_img_path="../notebooks/masks/aquarium/aquarium_mask.png"
        # mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        # mask = mask / 255

        # _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, mask=mask
        # )
        START=1
        np.save("/home/young/GenPose2/start.npy",START)
    else:
        rgb_image = camera.capture_image("rgb")
        depth_image = camera.capture_image("depth")
        # 保存图片
        cv2.imwrite(f"output_img/{count}_color.png", rgb_image)
        
        cv2.imwrite(f'output_img/{count}_depth.jpg', depth_image)
        out_obj_ids, out_mask_logits = predictor.track(frame)
        
        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        # print(all_mask.shape)

        # print("\n",len(out_obj_ids))
        # 生成一个全为 255 的背景
        all_mask2 = np.full((height, width, 1), 255, dtype=np.uint8)

        for i in range(0, len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                np.uint8
            ) * 255
            all_mask = cv2.bitwise_or(all_mask, out_mask)
            all_mask2[out_mask  > 0.0] = i
        # 取反
        # mask = cv2.bitwise_not(all_mask)
        # 将背景设置为 255
        cv2.imwrite(f'output_img/{count}_mask.png', all_mask2)
        count+=1
        
        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
        
        
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    

# cap.release()
# gif = imageio.mimsave("./result.gif", frame_list, "GIF", duration=0.00085)
