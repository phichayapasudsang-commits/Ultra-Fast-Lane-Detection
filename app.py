import os
import cv2
import torch
import numpy as np
import scipy.special
import gradio as gr

from model.model import parsingNet
from data.constant import culane_row_anchor

# ======================================
# 1. DEVICE AUTO SELECT
# ======================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ======================================
# 2. MODEL PATH
# ======================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "exp",
    "20260416_145707_lr_1e-04_b_8thai_finetune_v1",
    "ep009.pth"
)

# ======================================
# 3. MODEL CONFIG
# ======================================
griding_num = 200
cls_num_per_lane = 18

# ======================================
# 4. LOAD MODEL
# ======================================
model_ready = False

try:
    net = parsingNet(
        pretrained=False,
        backbone='18',
        cls_dim=(griding_num + 1, cls_num_per_lane, 4),
        use_aux=False
    ).to(device)

    if os.path.exists(MODEL_PATH):

        state_dict = torch.load(MODEL_PATH, map_location=device)

        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        new_state_dict = {}

        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        net.load_state_dict(new_state_dict, strict=False)
        net.eval()

        model_ready = True
        print("✅ Model loaded successfully")

    else:
        print("❌ Model file not found:")
        print(MODEL_PATH)

except Exception as e:
    print("❌ Error loading model:")
    print(e)

# ======================================
# 5. PROCESS IMAGE
# ======================================
def process_image(img_ori):

    if img_ori is None:
        return None

    if not model_ready:
        return img_ori

    h_ori, w_ori = img_ori.shape[:2]

    # Gradio image = RGB
    img = cv2.resize(img_ori, (800, 288))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = (img - mean) / std
    img = img.transpose(2, 0, 1)

    img = torch.from_numpy(img).float().unsqueeze(0).to(device)

    # =============================
    # INFERENCE
    # =============================
    with torch.no_grad():
        out = net(img)

    out = out[0].detach().cpu().numpy()
    out = out[:, ::-1, :]

    # =============================
    # DECODE
    # =============================
    prob = scipy.special.softmax(out[:-1, :, :], axis=0)

    idx = np.arange(1, griding_num + 1).reshape(-1, 1, 1)

    loc = np.sum(prob * idx, axis=0)

    out_argmax = np.argmax(out, axis=0)

    loc[out_argmax == griding_num] = 0

    # =============================
    # DRAW LANE
    # =============================
    col_sample = np.linspace(0, 799, griding_num)

    result_img = img_ori.copy()

    for lane_idx in range(loc.shape[1]):

        lane = loc[:, lane_idx]

        if np.sum(lane > 0) < 2:
            continue

        points = []

        for row_idx in range(len(lane)):

            val = lane[row_idx]

            if val > 0:

                index = int(round(val)) - 1

                if 0 <= index < griding_num:

                    x = int(col_sample[index] * w_ori / 800)

                    y = int(
                        h_ori *
                        (
                            culane_row_anchor[
                                cls_num_per_lane - 1 - row_idx
                            ] / 288
                        )
                    )

                    points.append((x, y))

        if len(points) > 2:
            pts = np.array(points, np.int32)

            cv2.polylines(
                result_img,
                [pts],
                False,
                (0, 255, 0),
                8
            )

    return result_img

# ======================================
# 6. GRADIO UI
# ======================================
demo = gr.Interface(
    fn=process_image,

    inputs=gr.Image(
        type="numpy",
        label="Upload Road Image"
    ),

    outputs=gr.Image(
        type="numpy",
        label="Lane Detection Result"
    ),

    title="🚗 Ultra Fast Lane Detection",
    description="Upload an image to test lane detection model",

    flagging_mode="never"
)

# ======================================
# 7. RUN
# ======================================
if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True
    )