from __future__ import print_function
import onnxruntime
import torch
import numpy as np
import tqdm
import cv2


device = torch.device("cpu")
#onnx路径
model_path = "GoToMb_640640.onnx"
#数据前处理
image_path = './TEST/messi2.jpg'
image_tensor = cv2.imread(image_path, cv2.IMREAD_COLOR)
img = cv2.resize(image_tensor, (640, 640), interpolation=cv2.INTER_LINEAR)
img = np.transpose(img,(2,0,1))
img = np.expand_dims(img, 0) #添加一个维度 就是batch维度
img = img.astype(np.float32)#格式转成float32
img /= 255.0
#加载onnx模型
# img = torch.rand(1, 3, 640, 640).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
timings = np.zeros((300, 1))
#
ort_session = onnxruntime.InferenceSession(model_path, providers=device)
print("Warming ...\n")
with torch.no_grad():
    for i in tqdm.tqdm(range(1001)):
        _ = ort_session.run(
                None,
                {"input0": img},
            )
print("Testing ...\n")
with torch.no_grad():
    for rep in tqdm.tqdm(range(300)):
        starter.record()
        _ = ort_session.run(
                None,
                {"input0": img},
            )
        ender.record()
        # load_t1 = time.time()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

    avg = timings.sum()/300
    print("\navg={}\n".format(avg))

