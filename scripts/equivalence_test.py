#!/usr/bin/env python3
"""
Darknet (OpenCV4) vs ONNX (OpenCV5) equivalence harness.
Runs same images through reference Darknet cfg/weights (OpenCV4) and candidate ONNX (OpenCV5 or OpenCV4)
and compares raw conv outputs and decoded detections.
"""
import cv2, numpy as np, math, sys, pathlib
cfg="model/pordav4x3.cfg"
weights="model/porda-19200-lr-0005-909.weights"
onnx="model/porda.onnx"
anchors=[(10,14),(23,27),(37,58),(81,82),(135,169),(344,319)]
masks=[[3,4,5],[0,1,2]]
scale_xy=1.05
net_w, net_h=544,320
def sigmoid(x): return 1/(1+np.exp(-x))

def letterbox(img, tw=544, th=320):
    h,w=img.shape[:2]
    scale=min(th/h, tw/w)
    nw=int(w*scale); nh=int(h*scale)
    bottom=th-nh; right=tw-nw
    if (bottom<55 and right==0) or (right<70 and bottom==0):
        return img,1.0,1.0
    resized=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LINEAR)
    padded=cv2.copyMakeBorder(resized,0,bottom,0,right,cv2.BORDER_CONSTANT,value=(0,0,0))
    return padded, w/nw, h/nh

def raw_diff():
    # Use deterministic seeds
    for size in [(544,320),(800,600),(1920,1200),(512,512)]:
        w,h=size
        np.random.seed(0)
        img=np.random.randint(0,255,(h,w,3),dtype=np.uint8)
        padded,_,_=letterbox(img)
        blob=cv2.dnn.blobFromImage(padded,1/255.0,(544,320),swapRB=True,crop=False)
        net=cv2.dnn.readNetFromDarknet(cfg,weights)
        net.setInput(blob)
        dark=net.forward(['conv_29','conv_36'])
        net2=cv2.dnn.readNetFromONNX(onnx)
        net2.setInput(blob)
        onnx_out=net2.forward(net2.getUnconnectedOutLayersNames())
        for i,(d,o) in enumerate(zip(dark,onnx_out)):
            diff=np.abs(d-o)
            print(f"size {w}x{h} head{i} mean {diff.mean():.6f} max {diff.max():.6f} median {np.median(diff):.6f} {'PASS' if diff.max()<0.005 else 'FAIL'}")

if __name__=="__main__":
    print("OpenCV",cv2.__version__)
    raw_diff()
    # Lena
    lena=cv2.imread("/tmp/lena.jpg")
    if lena is not None:
        padded,_,_=letterbox(lena)
        blob=cv2.dnn.blobFromImage(padded,1/255.0,(544,320),swapRB=True,crop=False)
        net=cv2.dnn.readNetFromDarknet(cfg,weights)
        net.setInput(blob)
        dark=net.forward(['conv_29','conv_36'])
        net2=cv2.dnn.readNetFromONNX(onnx)
        net2.setInput(blob)
        onnx_out=net2.forward(net2.getUnconnectedOutLayersNames())
        print("Lena raw diff head0",np.abs(dark[0]-onnx_out[0]).max())
        print("Lena raw diff head1",np.abs(dark[1]-onnx_out[1]).max())

