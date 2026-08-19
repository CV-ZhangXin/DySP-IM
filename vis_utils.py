import os
import cv2
import numpy as np
import random
import json
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from einops import rearrange

def get_cam_1d(classifier, feat,attention):
    attention = torch.nn.functional.softmax(attention,dim=-1)
    features = torch.einsum('ns,n->ns', feat.squeeze(0), attention)  ### n x fs
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('gf,cf->cg', features, tweight)
    cam_maps = torch.nn.functional.softmax(cam_maps, dim=0)
    return cam_maps[1]

def get_cam_1d_trans(classifier,feat,attention,to_out,norm):
    b,h,n,d = feat.size()

    features = torch.einsum('hnd,hn -> hnd', feat.squeeze(0), attention.squeeze(0))
    features = rearrange(features, 'h n d -> n (h d)', h = h)
    features = to_out[0](features)
    features = norm(features)

    cam_maps = classifier(features)
    cam_maps = torch.nn.functional.softmax(cam_maps,dim=-1)

    return cam_maps[:,1]

def get_area(pos_anchors,margin_percentage = 2,center_anchors=None,width_height=None):
    if center_anchors is not None:
        margin = 10 * 512 * margin_percentage / 2
        center_anchor = (center_anchors[1],center_anchors[0])
    else:
        top, down, left, right = min(pos_anchors[:, 1]), max(pos_anchors[:, 1]), min(pos_anchors[:, 0]), max(pos_anchors[:, 0])
        center_anchor = ((top + down) // 2, (left + right) // 2)
        margin = max((down - top), (right - left)) * margin_percentage / 2

    top, down, left, right = np.array(
        [center_anchor[0]-margin, center_anchor[0]+margin, center_anchor[1]-margin, center_anchor[1]+margin],
        dtype=int
    )
    ori_coord = np.array([down,top,right,left])
    top,down,left,right = np.clip(top,0,width_height[1]),np.clip(down,0,width_height[1]),np.clip(left,0,width_height[0]),np.clip(right,0,width_height[0])
    _gap = np.array([down,top,right,left]) 
    _coord = np.array([top,down,left,right]) 
    top,down,left,right = _coord + (_gap - ori_coord)

    return top,down,left,right,(down-top) * (right-left)
    

def screen_coords(scores, coords, top_left, bot_right,cam=None):
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    if cam is not None:
        return scores[mask], coords[mask], cam[mask]
    else:
        return scores[mask], coords[mask]


def random_area(width,height,min_area,max_area):
    while True:
        left = random.randint(0, width - 1)
        top = random.randint(0, height - 1)
        right = random.randint(left + 1, width)
        down = random.randint(top + 1, height)
        area = (right - left) * (down - top)
        if min_area <= area <= max_area:
            return (top, left, down, right)


def inference_pipeline(model, features, baseline='attn', need_mask=False, A_norm_type='minmax'):
    features = features.unsqueeze(0)

    # A: hard instances, cam: Top 1% score instances
    if need_mask:
            # mask predition
            pred, A = model.forward_teacher(features)
            pred = model.predictor(pred)
            # Convert A to a 1D tensor and sort it to find the threshold
            A_flat = A.view(-1)
            sorted_A, indices = torch.sort(A_flat, descending=True)
            # Calculate the threshold index for the top 1%
            threshold_index = int(0.01 * A_flat.size(0))
            # Get the threshold value
            threshold_value = sorted_A[threshold_index]
            # Create a mask for values greater than or equal to the threshold value
            cam = (A >= threshold_value).float().squeeze()

            # masked instances is 0, unmasked (hard instance) is 1
            len_keep,mask_ids = model.get_mask(features.size(1),0,A)
            A_mask = A.clone()
            A_mask[:] = 0
            _m = A_mask.clone()
            _m[:] = 0
            _m = _m.scatter_(1,mask_ids[:,:len_keep],1) == 1
            A_mask[_m] = 1
            # random mask
            # Get the indices of elements in A_mask that are 1
            indices = torch.nonzero(A_mask, as_tuple=False)
            # Calculate the number of elements to set to 0 (50% of the 1s)
            num_to_zero = int(indices.size(0) * model.merge.merge_ratio)
            # Randomly select indices to set to 0
            random_indices = indices[torch.randperm(indices.size(0))[:num_to_zero]]
            # Set the selected indices to 0
            A_mask[random_indices[:, 0], random_indices[:, 1]] = 0
            A = A_mask.squeeze()
    else:
        if baseline == 'attn':
            pred, A = model.forward_test(features, return_attn=True, return_act=True,no_norm=True)
            A,act = A
            cam = get_cam_1d(model.predictor, act, A.squeeze())
        elif baseline == 'selfattn':
            pred, A = model.forward_test(features, return_attn=True, return_act=True,no_norm=False)
            A,act = A
            
            _, A_nonorm = model.forward_test(features, return_attn=True, no_norm=True)
            cam = get_cam_1d_trans(model.predictor,act,A[0],model.online_encoder.layer1.attn.to_out,model.online_encoder.norm)
            # 多个head该怎么处理attn？先试试平均池化把
            A = torch.mean(A_nonorm[0].squeeze(),dim=0)
        elif baseline == 'dsmil':
            pred, A = model.forward_test(features, return_attn=True, return_cam=True,no_norm=True)
            pred,_ = pred
            A,cam = A
            cam = torch.nn.functional.softmax(cam.squeeze(),dim=-1)
            cam = cam[:,1]
        
    print(torch.nn.functional.softmax(pred))
    cam_thr = cam.max()
    print(cam_thr)

    # Normalize A
    A = A.squeeze()
    if A_norm_type == 'minmax':
        A = (A - A.min()) / (A.max() - A.min())
    else:
        A = torch.nn.functional.softmax(A,dim=-1)

    # Sort and visualize A
    _a, _a_idx = torch.sort(A, descending=True)
    sns.scatterplot(x=np.array(list(range(_a.size(0)))), y=_a)

    # Sort cam
    _cam, _cam_idx = torch.sort(cam, descending=True)
    sns.scatterplot(x=np.array(list(range(_cam.size(0)))), y=_cam)
    print(_cam[:3])

    # Return relevant variables
    return pred, A, cam, cam_thr, _a, _cam, _a_idx, _cam_idx


def find_roi_in_slide(label, width, height, patch_num=1, patch_size=512):
    if label is not None:
        for _pos in range(len(label['positive'])):
            _, _, _, _, area = get_area(np.array(label['positive'][_pos]['vertices']), width_height=[width, height])
            if area >= (patch_size * patch_num) ** 2:
                print(_pos, area ** 0.5 / patch_size)


def visualize_slide(slide, roi, vis_mode,coords, _A ,_cam , width, height, _a_idx,
                    f_id, f_name, label=None, vis_level=3, vis_size=None, alpha=0.3, 
                    margin_percentage=1.5, roi_coord=None, rel_norm=False, filter_thr=0.4, filter_thr_cam=0.5001, 
                    save_figure=False, _c_cam=np.array([0, 255, 255]), _c_attn=np.array([255, 255, 255]), 
                    crop_size=512, stride=512):

    if roi_coord is None:
        if roi >= 0:
            if label is not None:
                pos_anchors = np.array(label['positive'][roi]['vertices'])
                top, down, left, right, _ = get_area(pos_anchors, margin_percentage, width_height=[width, height])
            else:
                top, down, left, right, _ = get_area(None, margin_percentage, coords[_a_idx[0]], width_height=[width, height])
        else:
            top, down, left, right = 0, height, 0, width
        crop_coords = []
        for i in range(top, down, stride):
            for j in range(left, right, stride):
                if j + crop_size > width or i + crop_size > height:
                    continue
                crop_coords.append((j, i))
        right, down = np.max(crop_coords, 0) + crop_size
    else:
        top, down, left, right = roi_coord
    print(">>>>>>>>>>>> region coord gotten >>>>>>>>>>>")

    scale_level_ratio = np.array(slide.level_dimensions[vis_level]) / np.array(slide.level_dimensions[0])
    _w, _h = np.array((right-left, down-top)) * scale_level_ratio
    region = slide.read_region((left, top), list(range(slide.level_count))[vis_level], (int(_w), int(_h))).convert('RGB')
    print(">>>>>>>>>>>> slide region gotten >>>>>>>>>>>")

    if vis_mode == 'ori':
        if vis_size is not None:
            scale_ratio = (vis_size / img.shape[0])*scale_level_ratio
        else:
            scale_ratio = scale_level_ratio
        plt.imshow(np.array(region))
        plt.axis("off")
        plt.xlim(0, (right-left)*scale_ratio[0])
        plt.ylim((down-top)*scale_ratio[1], 0)
        if label is not None:
            for anchors in label['positive']:
                _pos_anchors = np.array(anchors['vertices'])
                plt.plot((_pos_anchors[:, 0] - left) * scale_ratio[0], (_pos_anchors[:, 1] - top) * scale_ratio[1], color='deepskyblue')
        if save_figure:
            path = os.path.join("./vis_figure/sup/", f_id + f_name + '_' + str(roi) + '_ori.png')
            plt.savefig(path, dpi=450, bbox_inches='tight')
        plt.show()
        return
    
    elif vis_mode == 'both':
        A_roi, coords_roi, cam_roi = screen_coords(_A, coords, (left, top), (right, down), cam=_cam)
        sorted_indices = np.argsort(cam_roi)
        print(cam_roi[sorted_indices[-5:]])
    elif vis_mode == 'cam':
        A_roi,_A = None,None
        cam_roi,coords_roi = screen_coords(_cam, coords, (left, top), (right, down))
    elif vis_mode == 'attn':
        cam_roi,_cam = None,None
        A_roi, coords_roi = screen_coords(_A, coords, (left, top), (right, down))
    else:
        raise ValueError("Invalid visualization mode")
    
    img = np.array(region)
    coords_roi_rel = []
    for i in range(len(coords_roi)):
        rel_x = coords_roi[i][0] - left
        rel_y = coords_roi[i][1] - top
        rel_x, rel_y = np.clip(rel_x, 0, right-left), np.clip(rel_y, 0, down-top)
        coords_roi_rel.append([rel_x, rel_y])
    print(">>>>>>>>>>>> relative coords computed >>>>>>>>>>>")

    # this only local norm
    if _A is not None:
        if rel_norm:
            A_roi = (A_roi - A_roi.min()) / (A_roi.max() - A_roi.min())
        # A_roi[A_roi < filter_thr] = 0
        A_roi = np.where(A_roi > filter_thr, A_roi - filter_thr + 0.4, 0)

    if _cam is not None:
        cam_roi[cam_roi <= filter_thr_cam] = 0
        # 为了颜色问题，这里对0.5以下，但又没有筛掉的部分做下限提升。主要是因为selfattn和dsmil做了norm之后的cam中间值变成了0.2左右，不再是0.5。该操作对ABMIL没有影响
        cam_roi[(cam_roi <= 0.5) & (cam_roi > 0)] = 0.5

    heatmap_attn = np.zeros(img.shape)
    heatmap_cam = np.zeros(img.shape)
    alpha_mat_attn = np.ones(img.shape) * 0.5 * alpha
    alpha_mat_cam = np.ones(img.shape) * 0.5 * alpha

    for _idx, _coord in enumerate(coords_roi_rel):
        A_flag = False
        cam_flag = False

        if _A is not None and A_roi[_idx] != 0:
            heatmap_attn[int(_coord[1]*scale_level_ratio[1]):int((_coord[1]+crop_size)*scale_level_ratio[1]),
                         int(_coord[0]*scale_level_ratio[0]):int((_coord[0]+crop_size)*scale_level_ratio[0]), :] = A_roi[_idx] * _c_attn
            A_flag = True
        if _cam is not None and cam_roi[_idx] != 0:
            alpha_mat_attn[int(_coord[1]*scale_level_ratio[1]):int((_coord[1]+crop_size)*scale_level_ratio[1]),
                           int(_coord[0]*scale_level_ratio[0]):int((_coord[0]+crop_size)*scale_level_ratio[0]), :] = 0
            heatmap_attn[int(_coord[1]*scale_level_ratio[1]):int((_coord[1]+crop_size)*scale_level_ratio[1]),
                         int(_coord[0]*scale_level_ratio[0]):int((_coord[0]+crop_size)*scale_level_ratio[0]), :] = 0
            alpha_mat_cam[int(_coord[1]*scale_level_ratio[1]):int((_coord[1]+crop_size)*scale_level_ratio[1]),
                          int(_coord[0]*scale_level_ratio[0]):int((_coord[0]+crop_size)*scale_level_ratio[0]), :] = 1 - cam_roi[_idx]
            heatmap_cam[int(_coord[1]*scale_level_ratio[1]):int((_coord[1]+crop_size)*scale_level_ratio[1]),
                        int(_coord[0]*scale_level_ratio[0]):int((_coord[0]+crop_size)*scale_level_ratio[0]), :] = _c_cam
            cam_flag = True
        else:
            alpha_mat_cam[int(_coord[1]*scale_level_ratio[1]):int((_coord[1]+crop_size)*scale_level_ratio[1]),
                          int(_coord[0]*scale_level_ratio[0]):int((_coord[0]+crop_size)*scale_level_ratio[0]), :] = 0

        if not A_flag and not cam_flag:
            alpha_mat_attn[int(_coord[1]*scale_level_ratio[1]):int((_coord[1]+crop_size)*scale_level_ratio[1]),
                           int(_coord[0]*scale_level_ratio[0]):int((_coord[0]+crop_size)*scale_level_ratio[0]), :] = alpha
    print(">>>>>>>>>>>> heatmap computed >>>>>>>>>>>")

    blended_image = (alpha_mat_attn * img + alpha_mat_cam * img + (1 - alpha_mat_attn) * heatmap_attn + (1 - alpha_mat_cam) * heatmap_cam)
    blended_image = blended_image.astype(np.uint8)

    if vis_size is not None:
        plt.imshow(cv2.resize(blended_image, dsize=(vis_size, vis_size)))
        scale_ratio = (vis_size / img.shape[0]) * scale_level_ratio
    else:
        plt.imshow(blended_image)
        scale_ratio = scale_level_ratio
    print(">>>>>>>>>>>> heatmap draw done>>>>>>>>>>>")
    plt.xlim(0, (right-left)*scale_ratio[0])
    plt.ylim((down-top)*scale_ratio[1], 0)
    plt.axis("off")

    if label is not None:
        for anchors in label['positive']:
            _pos_anchors = np.array(anchors['vertices'])
            plt.plot((_pos_anchors[:, 0] - left) * scale_ratio[0], (_pos_anchors[:, 1] - top) * scale_ratio[1], color=(0, 128/255, 1), linewidth=3.0)

    if save_figure:
        path = os.path.join("./vis_figure/sup/tumor/", f_id + f_name + str(roi) + '.png')
        plt.savefig(path, dpi=450, bbox_inches='tight')
    plt.show()

    