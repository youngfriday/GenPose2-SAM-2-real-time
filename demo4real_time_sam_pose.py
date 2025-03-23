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
from infer import create_genpose2, data_pre, InferDataset, visualize_pose
import sys
# sys.path.append('segment-anything-2-real-time')
from realsense_camera import RealSenseCamera
from segment_anything_2_real_time.grounding_dino.groundingdino.util.inference import load_model, load_image, predict
GROUNDING_DINO_CONFIG = "segment_anything_2_real_time/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "segment_anything_2_real_time/gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25



# TODO!!!!! change to your own text prompt
TEXT_PROMPT = "kettle , pen"

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from segment_anything_2_real_time.sam2.build_sam import build_sam2_camera_predictor
import time

# genpose2 model
######################################## PARAMETERS ########################################
TRACKING = True                      # Tracking mode

# Tracking parameter, if the relative pose between the current frame and the previous frame
# is large, such as low video FPS or fast object motion, you can set a larger value. The default
# TRACKING_TO is set to 0.15.
TRACKING_T0 = 0.15

SCORE_MODEL_PATH='results/ckpts/ScoreNet/scorenet.pth'     # Path to the score model
ENERGY_MODEL_PATH='results/ckpts/EnergyNet/energynet.pth'  # Path to the energy model
SCALE_MODEL_PATH='results/ckpts/ScaleNet/scalenet.pth'     # Path to the scale model
PREV_POSE = None                                           # Previous pose
######################################## PARAMETERS ########################################

GenPose2 = create_genpose2(
    score_model_path=SCORE_MODEL_PATH, 
    energy_model_path=ENERGY_MODEL_PATH,
    scale_model_path=SCALE_MODEL_PATH,
)
cv2.namedWindow('rgb')
######################################## PARAMETERS ########################################

# build grounding dino model from local path
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device="cuda"
)
sam2_checkpoint = "segment_anything_2_real_time/checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "/configs/sam2.1/sam2.1_hiera_b+.yaml"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# cap = cv2.VideoCapture(0)
camera = RealSenseCamera()
# sleep to prepare the camera
time.sleep(5)
if_init = False

# prompt grounding dino to get the box coordinates on specific frame

rgb_image = camera.capture_image("rgb")
# save the first scene image 
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

while True:
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
    
        cv2.imwrite(f'output_img/{count}_mask.png', all_mask2)
        
        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

        color_path = f'output_img/{count}_color.png'
        depth_path = f'output_img/{count}_depth.jpg'
        mask_path = f'output_img/{count}_mask.png'
        meta_path = 'assets/meta.json' 
        count+=1      
        print("count=",count) 
        data=data_pre(color_path, depth_path, mask_path, meta_path)
        data = InferDataset(data, img_size=GenPose2.cfg.img_size, device=GenPose2.cfg.device, n_pts=GenPose2.cfg.num_points)
        pose, length = GenPose2.inference(data, PREV_POSE, TRACKING, TRACKING_T0)
        color_image_w_pose = visualize_pose(data, pose, length, visualize_image=False)
        PREV_POSE = pose
        cv2.imshow('rgb', color_image_w_pose)
        cv2.waitKey(1) 
        # cv2.destroyAllWindows() 
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)    
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    

# cap.release()
# gif = imageio.mimsave("./result.gif", frame_list, "GIF", duration=0.00085)
