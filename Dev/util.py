import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# see https://blog.csdn.net/leviopku/article/details/82660381, resn
class conv_bn_relu(nn.Module):
    def __init__(self, nin, nout, kernel, stride=1, pad="SAME", padding=0, bn=True, activation="leakyRelu"):
        super().__init__()
        
        self.bn = bn
        self.activation = activation
        
        if pad == 'SAME':
            padding = (kernel-1)//2
            
        self.conv = nn.Conv2d(nin, nout, kernel, stride, padding, bias=not bn)
        if bn == True:
            self.bn = nn.BatchNorm2d(nout)
        if activation == "leakyRelu":
            self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class res_layer(nn.Module):
    def __init__(self, nin):
        super().__init__()
        
        self.conv1 = conv_bn_relu(nin, nin//2, kernel=1)  #64->32, 1
        self.conv2 = conv_bn_relu(nin//2, nin, kernel=3)  #32->64, 3 see figure of darknet
        
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) # just '+', the dim will be the same, not concat!
    
    
def map2cfgDict(mlist):
    idx = 0 
    mdict = OrderedDict()
    for i,m in enumerate(mlist):
        if isinstance(m, res_layer):
            mdict[idx] = None
            mdict[idx+1] = None
            idx += 2
        mdict[idx] = i
        idx += 1
    
    return mdict        


# UpsampleGroup: conv + upsample + concat 
# see https://blog.csdn.net/leviopku/article/details/82660381, DBL + 上采样 + concat
class UpsampleGroup(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.conv = conv_bn_relu(nin, nin//2, kernel=1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        
    def forward(self, route_head, route_tail):
        out = self.up(self.conv(route_head))
        return torch.cat((out, route_tail), 1) # concat, size: nin/2 + nin
    
def make_res_stack(nin, num_block):
    return nn.ModuleList([conv_bn_relu(nin, nin*2, 3, stride=2)] + [res_layer(nin*2) for n in range(num_block)])

class Darknet(nn.Module):
    def __init__(self, blkList, nout=32):
        super().__init__()
        
        self.mlist = nn.ModuleList()
        self.mlist += [conv_bn_relu(3, nout, 3)]
        for i,nb in enumerate(blkList):
            self.mlist += make_res_stack(nout*(2**i), nb)
            
        self.map2yolocfg = map2cfgDict(self.mlist)
        self.cachedOutDict = dict()
        
    def forward(self, x):
        for i,m in enumerate(self.mlist):
            x = m(x)
            if i in self.cachedOutDict:
                self.cachedOutDict[i] = x
        return x
    
    def addCachedOut(self, idx, mode="yolocfg"):
        if mode == "yolocfg":
            idxs = self.map2yolocfg[idx]
        self.cachedOutDict[idxs] = None
        
    def getCachedOut(self, idx, mode="yolocfg"):
        if mode == "yolocfg":
            idxs = self.map2yolocfg[idx]
        return self.cachedOutDict[idxs]
    
    
class PreDetectionConvGroup(nn.Module):
    def __init__(self, nin, nout, num_conv=3, numClass=80):
        super().__init__()
        
        self.mlist = nn.ModuleList()
        
        for i in range(num_conv): # 2*3 = 6,see figure of conv set on https://blog.csdn.net/qq_37541097/article/details/81214953 
            self.mlist += [conv_bn_relu(nin, nout, kernel=1)]
            self.mlist += [conv_bn_relu(nout, nout*2, kernel=3)]
            if i == 0:
                nin = nout*2
                
        self.mlist += [nn.Conv2d(nin, (numClass+5)*3, 1)] # expand dim to (numClass+5)*3
        self.map2yolocfg = map2cfgDict(self.mlist)
        self.cachedOutDict = dict()
        
    def forward(self, x):
        for i,m in enumerate(self.mlist):
            x = m(x)
            if i in self.cachedOutDict:
                self.cachedOutDict[i] = x
        
        return x
    
    #mode - normal  -- direct index to mlist 
    #     - yolocfg -- index follow the sequences of the cfg file from https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
    def addCachedOut(self, idx, mode="yolocfg"):
        if mode == "yolocfg":
            idx = self.getIdxFromYoloIdx(idx)
        elif idx < 0:
            idx = len(self.mlist) - idx
        
        self.cachedOutDict[idx] = None
        
    def getCachedOut(self, idx, mode="yolocfg"):
        if mode == "yolocfg":
            idx = self.getIdxFromYoloIdx(idx)
        elif idx < 0:
            idx = len(self.mlist) - idx
        return self.cachedOutDict[idx]
    
    def getIdxFromYoloIdx(self,idx):
        if idx < 0:
            return len(self.map2yolocfg) + idx
        else:
            return self.map2yolocfg[idx]