"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""
# 실행결과는 원본 이미지에 polygon 그리고  및 polygon 좌표 txt 파일 생성
# ai_hub_package_for_github/CRAFT_Make_Polygon/result 에 있음
# 원본 데이터는 'ai_hub_package_for_github/CRAFT_Make_Polygon/my_test_images'에 저장

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
from CRAFT_Make_Polygon import craft_utils
from CRAFT_Make_Polygon  import imgproc
from CRAFT_Make_Polygon  import file_utils
import json
import zipfile

from  CRAFT_Make_Polygon.craft  import CRAFT

from collections import OrderedDict
from utils.delete_file import delete_files_in_directory


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")






def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, opt.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=opt.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.to("cpu")

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if opt.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



if __name__ == '__main__':


    
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    #parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    #parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    #parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
# --- 옵션(Argument) 파서 ---
   
    # [✅ 수정] 추론용 LMDB 경로를 받도록 변경
    parser.add_argument('--trained_model', type=str, default ='./CRAFT_Make_Polygon/craft_mlt_25k.pth', help='path to CRAFT model')
    parser.add_argument('--test_folder', type=str, default='./CRAFT_Make_Polygon/my_test_images',  help='path to 원본 이미지 폴더')
    parser.add_argument('--result_folder', type=str, default='./CRAFT_Make_Polygon/result/',  help='path to 결과물 폴더')
    
         

    opt = parser.parse_args()
    
    
    delete_files_in_directory(opt.result_folder, ["*.jpg", "*.txt"])
   
    
    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(opt.test_folder)

    #result_folder = './result/'

    if not os.path.isdir(opt.result_folder):
       os.mkdir(opt.result_folder)


     # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + opt.trained_model + ')')
    
    if opt.cuda:
        net.load_state_dict(copyStateDict(torch.load(opt.trained_model, map_location='cpu')))
    else:
        net.load_state_dict(copyStateDict(torch.load(opt.trained_model, map_location='cpu')))

    if opt.cuda:
        net = net.to("cpu")
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

   
    net.eval()

    # LinkRefiner
    refine_net = None
    if opt.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + opt.refiner_model + ')')
        if opt.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(opt.refiner_model)))
            refine_net = refine_net.to("cpu")
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(opt.refiner_model, map_location='cpu')))

        refine_net.eval()
        opt.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, opt.text_threshold, opt.link_threshold, opt.low_text, opt.cuda, opt.poly, refine_net)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = opt.result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=opt.result_folder)

    print("elapsed time : {}s".format(time.time() - t))
