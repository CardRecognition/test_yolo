import onnxruntime as ort
import numpy as np
from PIL import Image

MODEL_PATH = "src\detector.onnx"
IMG_PATH = "src\\test2.jpg" 

sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
print("Inputs :", [i.name for i in sess.get_inputs()])
print("Outputs:", [o.name for o in sess.get_outputs()])

inp_name = sess.get_inputs()[0].name
out_names = [o.name for o in sess.get_outputs()]

img = Image.open(IMG_PATH).convert("RGB").resize((640, 640))
arr = np.array(img).astype(np.float32) / 255.0
arr = np.transpose(arr, (2, 0, 1))[None, :, :, :]

outs = sess.run(out_names, {inp_name: arr})

for name, out in zip(out_names, outs):
    print(f"\n=== OUTPUT {name} ===")
    print("shape:", out.shape)
    print("dtype:", out.dtype)
    flat = out.reshape(-1)
    print("first 24 values:", flat[:24])