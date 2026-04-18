# DATA
dataset='CULane'
data_root = '/content/drive/MyDrive/Comvision/Ultra-Fast-Lane-Detection/fine_tune_dataset/'

# TRAIN
epoch = 20
batch_size = 8
optimizer = 'SGD'  #['SGD','Adam']
learning_rate = 2e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos' #['multi', 'cos']
steps = None
gamma  = 0.1
warmup = 'linear'
warmup_iters = 150

# NETWORK
use_aux = False
griding_num = 200
backbone = '18'

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = 'thai_finetune_v1'

log_path = '/content/drive/MyDrive/Comvision/Ultra-Fast-Lane-Detection/exp/'

# FINETUNE or RESUME MODEL PATH
finetune = '/content/drive/MyDrive/Comvision/Ultra-Fast-Lane-Detection/Pretrained-Model/culane_18.pth'
resume = None

# TEST
test_model = None
test_work_dir = None

num_lanes = 4




