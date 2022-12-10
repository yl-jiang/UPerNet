import random
from functools import reduce
import cv2
import numpy as np
import math
from skimage.transform import resize as skresize


__all__ = ["letter_resize", "RandomHSV", "RandomPerspective", "RandomBlur", 
          "RandomSaturation", "RandomBrightness",  "RandomCrop", "RandomFlipLR", 
          "RandomFlipUD", "RandomCutout", "resize_segmentation", "RandomShift"]


def letter_resize(img, seg, dst_size, stride=64, fill_value=114, only_ds=False, training=True):
    """
    only scale down
    Inputs:
        only_ds: only downsample
        fill_value: scalar
        training: bool
        img: (h, w, 3)
        seg: (h, w, 1)
        dst_size: int or [h, w]
        stride:
    Outputs:

    """
    if isinstance(dst_size, int):
        dst_size = [dst_size, dst_size]

    # 将dst_size调整到是stride的整数倍
    dst_del_h, dst_del_w = np.remainder(dst_size[0], stride), np.remainder(dst_size[1], stride)
    dst_pad_h = stride - dst_del_h if dst_del_h > 0 else 0
    dst_pad_w = stride - dst_del_w if dst_del_w > 0 else 0
    dst_size = [dst_size[0] + dst_pad_h, dst_size[1] + dst_pad_w]

    org_h, org_w = img.shape[:2]  # [height, width]
    scale = float(np.min([dst_size[0] / org_h, dst_size[1] / org_w]))
    if only_ds:
        scale = min(scale, 1.0)  # only scale down for good test performance

    resized_img = img.copy()
    resized_seg = seg.copy()
    if scale != 1.:
        resize_h, resize_w = int(org_h * scale), int(org_w * scale)
        resized_img = np.ascontiguousarray(cv2.resize(resized_img, (resize_w, resize_h), interpolation=4))
        resized_seg = np.ascontiguousarray(resize_segmentation(resized_seg[:, :, 0], (resize_h, resize_w)))
        if resized_seg.ndim == 2:
            resized_seg = resized_seg[..., None]
    else:
        resize_h, resize_w = img.shape[:2]

    img_out = np.full(shape=dst_size+[3], fill_value=fill_value)
    seg_out = np.full(shape=dst_size+[1], fill_value=0)
    # training时需要一个batch保持固定的尺寸，testing时尽可能少的填充像素以加速inference
    if not training:
        pad_h, pad_w = dst_size[0] - resize_h, dst_size[1] - resize_w
        pad_h, pad_w = np.remainder(pad_h, stride), np.remainder(pad_w, stride)
        top  = int(round(pad_h / 2))
        left = int(round(pad_w / 2))
        bottom = pad_h - top
        right  = pad_w - left
        if isinstance(fill_value, int):
            fill_value = (fill_value, fill_value, fill_value)
        img_out = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_value)
        seg_out = cv2.copyMakeBorder(resized_seg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    else:
        pad_h, pad_w  = dst_size[0] - resize_h, dst_size[1] - resize_w
        top, left     = pad_h // 2, pad_w // 2
        bottom, right = pad_h - top, pad_w - left
        img_out[top:(top+resize_h), left:(left+resize_w)] = resized_img
        seg_out[top:(top+resize_h), left:(left+resize_w)] = resized_seg
    return img_out.astype(np.uint8), seg_out.astype(np.uint8)

def min_scale_resize(img, dst_size):
    """

    :param img: ndarray
    :param dst_size: [h, w]
    :returns: resized_img, img_org
    """
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3
    min_scale = min(np.array(dst_size) / max(img.shape[:2]))
    h, w = img.shape[:2]
    if min_scale != 1:
        img_out = cv2.resize(img, (int(w*min_scale), int(h*min_scale)), interpolation=cv2.INTER_AREA)
        return img_out, img.shape[:2]
    return img

def RandomScale(img, seg, prob):
    """

    :param img: ndarray
    :param seg:
    :param thresh:
    :return:
    """
    assert isinstance(img, np.ndarray), f"Unkown Image Type {type(img)}"

    if random.random() < prob:
        scale = random.uniform(0.8, 1.2)
        # cv2.resize(img, shape)/其中shape->[宽，高]
        img_out = cv2.resize(img, None, fx=scale, fy=scale)
        seg_out = cv2.resize(seg, None, fx=scale, fy=scale)
        if seg_out.ndim == 2:
            seg_out = seg_out[..., None]
        return img_out, seg_out
    return img, seg

def RandomBlur(img, prob):
    """

    :param img: ndarray
    :param prob:
    :return:
    """
    assert isinstance(img, np.ndarray), f"Unkown Image Type {type(img)}"
    # 均值滤波平滑图像
    if random.random() < prob:
        img_out = cv2.blur(img, (5, 5))
        return img_out
    return img

def RandomSaturation(img, prob):
    # 图片饱和度
    if random.random() < prob:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        adjust = float(random.uniform(0.5, 1.5))
        s = adjust * s.astype(np.float32)
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        img_out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img_out
    return img

def RandomBrightness(img, prob):
    # 图片亮度
    if random.random() < prob:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # hsv分别表示：色调（H），饱和度（S），明度（V）
        h, s, v = cv2.split(hsv)
        adjust = float(random.uniform(0.8, 1.2))
        v = adjust * v.astype(np.float32)
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        img_out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img_out
    return img

def RandomShift(img, seg, prob, fill_value=114):
    # 随机平移
    if random.random() < prob:
        h, w, c = img.shape
        after_shfit_img = np.zeros_like(img)
        after_shfit_seg = np.zeros_like(seg)
        # after_shfit_image每行元素都设为[104,117,123]
        after_shfit_img[:, :, :] = (fill_value, fill_value, fill_value)  # bgr
        after_shfit_seg[:, :, :] = 0.0  # bgr
        shift_x = int(random.uniform(-w * 0.2, w * 0.2))
        shift_y = int(random.uniform(-h * 0.2, h * 0.2))
        # 图像平移
        if shift_x >= 0 and shift_y >= 0:  # 向下向右平移
            after_shfit_img[shift_y:, shift_x:, :] = img[:h - shift_y, :w - shift_x, :]
            after_shfit_seg[shift_y:, shift_x:, :] = seg[:h - shift_y, :w - shift_x, :]
        elif shift_x >= 0 and shift_y < 0:  # 向上向右平移
            after_shfit_img[:h + shift_y, shift_x:, :] = img[-shift_y:, :w - shift_x, :]
            after_shfit_seg[:h + shift_y, shift_x:, :] = seg[-shift_y:, :w - shift_x, :]
        elif shift_x <= 0 and shift_y >= 0:  # 向下向左平移
            after_shfit_img[shift_y:, :w + shift_x, :] = img[:h - shift_y, -shift_x:, :]
            after_shfit_seg[shift_y:, :w + shift_x, :] = seg[:h - shift_y, -shift_x:, :]
        else:  # 向上向左平移
            after_shfit_img[:h + shift_y, :w + shift_x, :] = img[-shift_y:, -shift_x:, :]
            after_shfit_seg[:h + shift_y, :w + shift_x, :] = seg[-shift_y:, -shift_x:, :]
        return after_shfit_img, after_shfit_seg
    return img, seg
        
def RandomCrop(img, seg, prob):
    # 随机裁剪
    if random.random() < prob:
        height, width, c = img.shape
        # x,y代表裁剪后的图像的中心坐标，h,w表示裁剪后的图像的高，宽
        h = random.uniform(0.6 * height, height)
        w = random.uniform(0.6 * width, width)
        x = random.uniform(width / 4, 3 * width / 4)
        y = random.uniform(height / 4, 3 * height / 4)

        # 左上角
        crop_xmin = np.clip(x - (w / 2), a_min=0, a_max=width).astype(np.int32)
        crop_ymin = np.clip(y - (h / 2), a_min=0, a_max=height).astype(np.int32)
        # 右下角
        crop_xmax = np.clip(x + (w / 2), a_min=0, a_max=width).astype(np.int32)
        crop_ymax = np.clip(y + (h / 2), a_min=0, a_max=height).astype(np.int32)

        cropped_img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]
        cropped_seg = seg[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]
        return cropped_img, cropped_seg
    return img, seg

def RandomHSV(img, prob, hgain=0.5, sgain=0.5, vgain=0.5):
    if random.random() < prob:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # no return needed
        return img_hsv
    return img

def RandomPerspective(img:np.ndarray, seg:np.ndarray, prob,
                      degree=10, 
                      translate=0.1, 
                      scale=0.1, 
                      shear=10,
                      perspective=0.0, 
                      dst_size=448, 
                      fill_value=114):
    """
    random perspective one image
    :param img:
    :param seg:  / ndarray
    :param degrees:
    :param translate:
    :param scale:
    :param shear:
    :param perspective:
    :param dst_size: output image size
    :param fill_value:
    :return:
    """
    if random.random() < prob:
        assert isinstance(img, np.ndarray)
        assert isinstance(seg, np.ndarray)
        assert img.ndim == 3, "only process one image once for now"
        assert img.shape[:2] == seg.shape[:2], f"image and seg's resilution should be the same, but got image's shape: {img.shape[:2]} seg's shape: {seg.shape[:2]}"

        if isinstance(dst_size, int):
            dst_size = [dst_size, dst_size]

        if img.shape[0] != dst_size:
            height, width = dst_size

        # Center / Translation / 平移
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective / 透视变换
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

        # Rotation and Scale / 旋转,缩放
        R = np.eye(3)
        a = random.uniform(-degree, degree)
        s = random.uniform(1 - scale, 1 + scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear / 剪切
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation / 平移
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (dst_size[0] != img.shape[0]) or (M != np.eye(3)).any():  # image changed
            if isinstance(fill_value, (int, tuple)):
                fill_value = (fill_value, fill_value, fill_value)
            if perspective:  # 是否进行透视变换
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=fill_value)
                seg = cv2.warpPerspective(seg, M, dsize=(width, height), borderValue=fill_value)
            else:  # 只进行仿射变换
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=fill_value)
                seg = cv2.warpAffine(seg, M[:2], dsize=(width, height), borderValue=fill_value)
        if seg.ndim == 2:
            seg = seg[..., None]
    return img, seg

def RandomFlipLR(img, seg, prob):
    """
    随机左右翻转image

    :param img: ndarray
    :param seg: ndarray
    :param prob:
    :return:
    """
    assert isinstance(img, np.ndarray), f"Unkown Image Type {type(img)}"

    # 水平翻转/y坐标不变,x坐标变化
    if random.random() < prob:
        img_out = np.fliplr(img)
        seg_out = np.fliplr(seg)
        return img_out, seg_out
    return img, seg

def RandomFlipUD(img, seg, prob):
    """
    随机上下翻转image。

    :param img: ndarray
    :param seg: 
    :param prob: 0 ~ 1
    """
    assert isinstance(img, np.ndarray), f"Unkown Image Type {type(img)}"

    # 竖直翻转/x坐标不变,y坐标变化
    if random.random() < prob:
        img_out = np.flipud(img)
        seg_out = np.flipud(seg)
        return img_out, seg_out
    return img, seg

def RandomCutout(img, seg, prob):
    """
    在图像中挖孔,并使用随机颜色填充。

    :param img: ndarray / (h, w, 3)
    :param seg: ndarray 
    :param cutout_p: 使用cutout的概率
    """
    if random.random() < prob:
        scales = [0.35] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        h, w, _ = img.shape
        if len(seg[seg > 0]) == 0:
            return img, seg
        img_cut, seg_cut = img.copy(), seg / seg.max()
        for s in scales:
            mask_h = np.random.randint(1, int(h * s))
            mask_w = np.random.randint(1, int(w * s))

            mask_xc = np.random.randint(0, w)
            mask_yc = np.random.randint(0, h)
            mask_xmin = np.clip(mask_xc - mask_w // 2, 0, w)  # (1,)
            mask_ymin = np.clip(mask_yc - mask_h // 2, 0, h)  # (1,)
            mask_xmax = np.clip(mask_xc + mask_w // 2, 0, w)  # (1,)
            mask_ymax = np.clip(mask_yc + mask_h // 2, 0, h)  # (1,)

            mask_w = mask_xmax - mask_xmin
            mask_h = mask_ymax - mask_ymin

            seg_fg = np.zeros_like(seg_cut)
            np.putmask(seg_fg, mask=seg_cut >= 0.5, values=[1])
            seg_fg_num = len(seg_cut[seg_cut > 0.5])
            if seg_fg_num == 0:
                continue
            ints_fg = seg_fg[mask_ymin:mask_ymax, mask_xmin:mask_xmax]
            ints_fg_num = len(ints_fg[ints_fg>0])
            ratio = ints_fg_num / seg_fg_num
            if ratio > 0.2:
                continue
            else:
                mask_color = [np.random.randint(69, 200) for _ in range(3)]
                img_cut[mask_ymin:mask_ymax, mask_xmin:mask_xmax] = mask_color
        return img_cut, seg
    return img, seg



def scale_jitting(img, seg, resize_shape=[320, 320] , dst_size=[224, 224]):
    """
    
    :param: dst_size: (h, w)
    将输入的image进行缩放后, 再从缩放后的image中裁剪固定尺寸的image
    """
    assert img.ndim == 3
    FLIP_LR = np.random.rand() > 0.5

    if dst_size and isinstance(dst_size, int):
        dst_size = [dst_size, dst_size]
    else:
        dst_size = img.shape[:2]

    resized_img = cv2.resize(np.ascontiguousarray(img.copy()), (resize_shape[0], resize_shape[1]), interpolation=cv2.INTER_LINEAR)
    resized_seg = cv2.resize(np.ascontiguousarray(seg.copy()), (resize_shape[0], resize_shape[1]), interpolation=cv2.INTER_LINEAR)
    
    hmin = np.random.randint(low=0, high=resize_shape[0], size=1)
    hmax = dst_size[0] + hmin
    wmin = np.random.randint(low=0, high=resize_shape[1], size=1)
    wmax = dst_size[1] + wmin

    if FLIP_LR:
        resized_img = resized_img[:, ::-1, :]
        out_img = resized_img[hmin:hmax, wmin:wmax, :]
        out_seg = resized_seg[hmin:hmax, wmin:wmax, :]
    
    return out_img, out_seg


def resize_segmentation(segmentation, new_shape, order=3, cval=0):
    '''
    copy from: https://github.com/MIC-DKFZ/nnUNet
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation: (H, W)
    :param new_shape: [h, w]
    :param order: 3 is Bi-cubic
        0: Nearest-neighbor
        1: Bi-linear (default)
        2: Bi-quadratic
        3: Bi-cubic
        4: Bi-quartic
        5: Bi-quintic
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return skresize(segmentation.astype(float), new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = skresize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped



if __name__ == '__main__':
    # test CV2Transform
    import matplotlib.pyplot as plt
    img_path = r'../../../../../Dataset/SOD/DUTS-TR-master/DUTS-TR-Image/ILSVRC2012_test_00000004.jpg'
    lab_path = r"../../../../../Dataset/SOD/DUTS-TR-master/DUTS-TR-Mask/ILSVRC2012_test_00000004.jpg"
    bbox_head = np.array([[96, 374, 143, 413]])
    cv_img = cv2.imread(img_path)
    cv_lab = cv2.imread(lab_path, 0)
    s = cv_img.shape
    print(cv_img.shape)
    

    


