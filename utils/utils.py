import torch
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def letterbox(img, height=416, color=(127.5, 127.5, 127.5)):
    # Resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh

def model_info(model):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %40s %9s %12g %20s %10.3g %10.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients' % (i + 1, n_p, n_g))
    
def plot_images(imgs, targets, fname='images.jpg'):
    # Plots training images overlaid with targets
    imgs = imgs.cpu().numpy()
    targets = targets.cpu().numpy()

    fig = plt.figure(figsize=(10, 10))
    img_size = imgs.shape[3]
    bs = imgs.shape[0]  # batch size
    sp = np.ceil(bs ** 0.5)  # subplots

    for i in range(bs):
        boxes = xywh2xyxy(targets[targets[:, 0] == i, 2:6]).T * img_size
        plt.subplot(sp, sp, i + 1).imshow(imgs[i].transpose(1, 2, 0))
        plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], '.-')
        plt.axis('off')
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close()
    
def build_targets(model, targets):
    # targets = [image, class, x, y, w, h]
    if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel):
        model = model.module

    nt = len(targets)
    txy, twh, tcls, indices = [], [], [], []
    for i in model.yolo_layers:
        layer = model.module_list[i][0]

        # iou of targets-anchors
        t, a = targets, []
        gwh = targets[:, 4:6] * layer.nG
        if nt:
            iou = [wh_iou(x, gwh) for x in layer.anchor_vec]
            iou, a = torch.stack(iou, 0).max(0)  # best iou and anchor

            # reject below threshold ious (OPTIONAL, increases P, lowers R)
            reject = True
            if reject:
                j = iou > 0.10
                t, a, gwh = targets[j], a[j], gwh[j]

        # Indices
        b, c = t[:, :2].long().t()  # target image, class
        gxy = t[:, 2:4] * layer.nG
        gi, gj = gxy.long().t()  # grid_i, grid_j
        indices.append((b, a, gj, gi))

        # XY coordinates
        txy.append(gxy - gxy.floor())

        # Width and height
        twh.append(torch.log(gwh / layer.anchor_vec[a]))  # wh yolo method
        # twh.append((gwh / layer.anchor_vec[a]) ** (1 / 3) / 2)  # wh power method

        # Class
        tcls.append(c)
        if c.shape[0]:
            assert c.max() <= layer.nC, 'Target classes exceed model classes'

    return txy, twh, tcls, indices


def compute_loss(p, targets):  # predictions, targets
    FT = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
    lxy, lwh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0])
    txy, twh, tcls, indices = targets
    MSE = nn.MSELoss()
    CE = nn.CrossEntropyLoss()
    BCE = nn.BCEWithLogitsLoss()

    # Compute losses
    # gp = [x.numel() for x in tconf]  # grid points
    for i, pi0 in enumerate(p):  # layer i predictions, i
        b, a, gj, gi = indices[i]  # image, anchor, gridx, gridy
        tconf = torch.zeros_like(pi0[..., 0])  # conf
        nt = len(b)  # number of targets

        # Compute losses
        k = 1  # nt / bs
        if nt:
            pi = pi0[b, a, gj, gi]  # predictions closest to anchors
            tconf[b, a, gj, gi] = 1  # conf

            lxy += (k * 8) * MSE(torch.sigmoid(pi[..., 0:2]), txy[i])  # xy loss
            lwh += (k * 1) * MSE(pi[..., 2:4], twh[i])  # wh yolo loss
            # lwh += (k * 1) * MSE(torch.sigmoid(pi[..., 2:4]), twh[i])  # wh power loss
            lcls += (k * 1) * CE(pi[..., 5:], tcls[i])  # class_conf loss

        # pos_weight = FT([gp[i] / min(gp) * 4.])
        # BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        lconf += (k * 64) * BCE(pi0[..., 4], tconf)  # obj_conf loss
    loss = lxy + lwh + lconf + lcls

    return loss, torch.cat((lxy, lwh, lconf, lcls, loss)).detach()