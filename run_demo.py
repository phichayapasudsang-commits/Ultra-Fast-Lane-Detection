import torch, cv2, os
import numpy as np
import scipy.special
from model.model import parsingNet
from data.constant import culane_row_anchor
from google.colab.patches import cv2_imshow

# ===============================
# 1. CONFIG (ตั้งค่าโฟลเดอร์)
# ===============================
MODEL_PATH = '/content/drive/MyDrive/Comvision/Ultra-Fast-Lane-Detection/exp/20260416_145707_lr_1e-04_b_8thai_finetune_v1/ep009.pth'

IMG_DIR    = "/content/drive/MyDrive/Comvision/Ultra-Fast-Lane-Detection/fine_tune_dataset/val_image/"
SAVE_DIR   = "/content/pred_images/"
PRED_DIR   = "/content/pred_txts/"

griding_num = 200
cls_num_per_lane = 18

# สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# ===============================
# 2. LOAD MODEL (โหลดโมเดลแค่ครั้งเดียว)
# ===============================
net = parsingNet(
    pretrained=False,
    backbone='18',
    cls_dim=(griding_num+1, cls_num_per_lane, 4),
    use_aux=False
).cuda()

state_dict = torch.load(MODEL_PATH, map_location='cpu')

if 'model' in state_dict:
    state_dict = state_dict['model']
elif 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v

load_info = net.load_state_dict(new_state_dict, strict=False)
net.eval()

print("✅ Model loaded check finished.")
print(f"⚠️ Missing keys: {len(load_info.missing_keys)}")
if len(load_info.missing_keys) > 0:
    print("🚨 ระวัง! โหลด Weights ไม่ครบ โมเดลอาจจะกำลังสุ่มค่า (ตรวจสอบไฟล์ .pth)")

# ===============================
# 3. PROCESS FOLDER (วนลูปอ่านรูปทั้งหมด)
# ===============================
valid_extensions = ('.jpg', '.jpeg', '.png')
img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(valid_extensions)]

print(f"📸 พบรูปภาพทั้งหมด {len(img_files)} รูป กำลังเริ่มประมวลผล...")

for img_name in img_files:
    img_path = os.path.join(IMG_DIR, img_name)
    save_path = os.path.join(SAVE_DIR, img_name)

    # --- LOAD IMAGE + PREPROCESS ---
    img_ori = cv2.imread(img_path)
    if img_ori is None:
        print(f"❌ อ่านไฟล์ไม่ได้ ข้าม: {img_name}")
        continue
        
    h_ori, w_ori = img_ori.shape[:2]
    img_rgb = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (800, 288))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float().unsqueeze(0).cuda()

    # --- INFERENCE ---
    with torch.no_grad():
        out = net(img)

    out = out[0].detach().cpu().numpy().copy()
    out = out[:, ::-1, :]

    # --- DECODE ---
    prob = scipy.special.softmax(out[:-1, :, :], axis=0)
    idx = np.arange(1, griding_num + 1).reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_argmax = np.argmax(out, axis=0)   
    loc[out_argmax == griding_num] = 0

    # --- DRAW LANE & PREPARE TXT ---
    col_sample = np.linspace(0, 800 - 1, griding_num)
    
    all_lanes_points = [] # 🔥 ลิสต์สำหรับเก็บพิกัดของทุกเลนในรูปนี้

    for lane_idx in range(loc.shape[1]):
        lane = loc[:, lane_idx]

        if np.sum(lane > 0) < 2:
            continue

        points = []
        for row_idx in range(len(lane)):
            val = lane[row_idx]
            
            if val > 0:
                index = int(np.round(val)) - 1
                
                if 0 <= index < griding_num:
                    x = int(col_sample[index] * w_ori / 800.0)
                    y = int(h_ori * (culane_row_anchor[cls_num_per_lane - 1 - row_idx] / 288.0))
                    points.append((x, y))

        if len(points) > 2:
            pts = np.array(points, np.int32)
            cv2.polylines(img_ori, [pts], False, (0, 255, 0), 10)
            
            # 🔥 เก็บพิกัดเลนนี้ไว้เพื่อนำไปเขียนลงไฟล์ .txt
            all_lanes_points.append(points)

    # --- SAVE RESULT IMAGE ---
    cv2.imwrite(save_path, img_ori)
    
    # --- 🔥 SAVE RESULT TXT (สำหรับ Evaluation) ---
    # เปลี่ยนนามสกุลไฟล์จาก .jpg/.png เป็น .txt
    txt_filename = os.path.splitext(img_name)[0] + '.txt'
    txt_save_path = os.path.join(PRED_DIR, txt_filename)
    
    with open(txt_save_path, 'w') as f:
        for lane_pts in all_lanes_points:
            # นำพิกัด (x, y) มาต่อกันคั่นด้วยช่องว่าง
            line_str = " ".join([f"{int(x)} {int(y)}" for x, y in lane_pts])
            f.write(line_str + "\n")

    print(f"✅ ประมวลผลและบันทึกภาพ+txt สำเร็จ: {img_name}")

print(f"\n🎉 เสร็จสมบูรณ์!")
print(f"รูปภาพเซฟที่: {SAVE_DIR}")
print(f"ไฟล์พิกัด .txt เซฟที่: {PRED_DIR} (พร้อมเอาไปรันโค้ดประเมินผล IoU แล้วครับ!)")