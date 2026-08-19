import openslide
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import cv2
import  os
from sklearn.manifold import TSNE

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



def get_cam_1d(classifier, feat,attention):
    attention = torch.nn.functional.softmax(attention)
    features = torch.einsum('ns,n->ns', feat, attention)  ### n x fs
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('gf,cf->cg', features, tweight)
    return cam_maps


# f_id:tumor_005,

def init_slide_info(f_id,tif_dir,json_dir,h5_dir):
    tif_path = os.path.join(tif_dir,f_id+".tif")
    json_path =  os.path.join(json_dir,f_id+".json")
    h5_path =    os.path.join( h5_dir,f_id+".h5")
    slide = openslide.OpenSlide(tif_path)
    width, height = slide.dimensions
    # load anno
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    except:
        json_data = None
        print('Normal slide')
    patch = h5py.File(h5_path,"r")
    features = torch.Tensor(patch['features'])
    coords = np.array(patch['coords'])

    return width,height,json_data,patch,features,coords,slide



def darw_original_img(f_id,tif_dir,json_dir,h5_dir):

    width, height, json_data, patch, features, coords,slide = init_slide_info(f_id,tif_dir,json_dir,h5_dir)
    plt.figure()
    crop_size = 512  # 图块大小
    stride = 512  # 步长
    vis_level = 3          # 可视化slide 的level，这个决定了坐标的计算level，最好0，其它level坐标有偏移
    vis_size = None # 448   #最后可视化图的大小
    roi = 0 # -1->global, i->roi_index, roi_index 对应上一个代码块的输出，对tumor slide有用
    alpha = .4  # 0-no_img,1-all_img
    margin_percentage = 2  #2 可以放大图像
    #roi_coord = (94500,157000,0,width)  #
    roi_coord = None  # 指定一个roi区域，一般不用 指定(top,down,left,right)，覆盖roi参数, None就按照roi参数执行
    rel_norm = False  #是否在roi区域范围内进行min-max正则化
    filter_thr = 0.75   # attention的阈值，一般0.5比较好
    filter_thr_cam = 0.5001 # cam_thr #cam的阈值，默认是mask模型cam的最大值，就可以屏蔽mask模型的病害图块，但要看这个值是不是太离谱，如果靠近0.5就可以，也要看ab-mil出来的效果，如果ab-mil病害太少，可以调小点
    is_norm = True    #默认为True，是否对attention进行max-min正则

    _c_cam = np.array([0,255,255])  # 这里可以改cam的颜色，rgb
    _c_attn = np.array([255,255,255]) # attn 颜色, 默认白色


    # 下面的代码是画原图的-----------------------------------------------------------------------------------
    if roi_coord is None:
        if roi >= 0:
            pos_anchors = np.array(json_data['positive'][roi]['vertices'])
            top,down,left,right,_ = get_area(pos_anchors,margin_percentage,width_height=[width,height])
        else:
            top,down,left,right = 0,height,0,width
        crop_coords = []
        # print(top,down,left,right)
        for i in range(top, down, stride):
            for j in range(left, right, stride):
                if j + crop_size > width or i + crop_size > height:
                    continue
                crop_coords.append((j, i))
        right, down = np.max(crop_coords, 0) + crop_size
    else:
        top,down,left,right = roi_coord
    print(">>>>>>>>>>>> region coord gotten >>>>>>>>>>>")

    scale_level_ratio = np.array(slide.level_dimensions[vis_level]) / np.array(slide.level_dimensions[0])
    _w,_h = np.array((right-left, down-top))*scale_level_ratio
    # _w,_h = (right-left, down-top)
    region = slide.read_region((left, top), list(range(slide.level_count))[vis_level], (int(_w),int(_h))).convert('RGB')

    img = np.array(region)
    # plt.imshow(cv2.resize(np.array(region),dsize=(4480,4480)))
    if vis_size is not None:
        plt.imshow(cv2.resize(img,dsize=(vis_size,vis_size)))
        scale_ratio = (vis_size / img.shape[0])*scale_level_ratio
    else:
        plt.imshow(img)
        scale_ratio = scale_level_ratio
    print(">>>>>>>>>>>> heatmap draw done>>>>>>>>>>>")
    plt.xlim(0,(right-left)*scale_ratio[0])
    plt.ylim((down-top)*scale_ratio[1],0)
    plt.axis("off")

    if json_data is not None:
        for anchors in (json_data['positive']):
            _pos_anchors = np.array(anchors['vertices'])
            plt.plot((_pos_anchors[:, 0]-left) * scale_ratio[0] , (_pos_anchors[:,1]-top)* scale_ratio[1], color=(0,128/255,1),linewidth=1.0)
    path = os.path.join("./img",f_id+'original.png')
    plt.savefig(path,dpi=450,bbox_inches='tight')
    plt.close()


"""
下面是画attention关注的点的-------------------------------------------------------------------------------------------------------------------------------------------------------
"""
def draw_attention(f_id,tif_dir,json_dir,h5_dir,classifier,A_b,level,feature):
    width, height, json_data, patch, features, coords, slide = init_slide_info(f_id, tif_dir, json_dir, h5_dir)

    plt.figure()

    crop_size = 512  # 图块大小
    stride = 512  # 步长
    vis_level = 3  # 可视化slide 的level，这个决定了坐标的计算level，最好0，其它level坐标有偏移
    vis_size = None  # 448   #最后可视化图的大小
    roi = 0  # -1->global, i->roi_index, roi_index 对应上一个代码块的输出，对tumor slide有用
    alpha = .4  # 0-no_img,1-all_img
    margin_percentage = 2  # 2 太大了内存吃不消，会卡死，这个主要看你roi的区域大小，roi太小这个就要大点，roi大了这个没必要弄得很大
    # roi_coord = (94500,157000,0,width)  #
    roi_coord = None  # 指定一个roi区域，一般不用 指定(top,down,left,right)，覆盖roi参数, None就按照roi参数执行
    rel_norm = False  # 是否在roi区域范围内进行min-max正则化
    filter_thr = 0.75  # attention的阈值，一般0.5比较好
    filter_thr_cam = 0.5001  # cam_thr #cam的阈值，默认是mask模型cam的最大值，就可以屏蔽mask模型的病害图块，但要看这个值是不是太离谱，如果靠近0.5就可以，也要看ab-mil出来的效果，如果ab-mil病害太少，可以调小点
    is_norm = True  # 默认为True，是否对attention进行max-min正则

    _c_cam = np.array([0, 255, 255])  # 这里可以改cam的颜色，rgb
    _c_attn = np.array([255, 255, 255])  # attn 颜色, 默认白色
    A_b = A_b.squeeze()
    _a,_a_idx = torch.sort(A_b,descending=True)


    cam_b = get_cam_1d(classifier,feature,A_b)
    cam_b = torch.nn.functional.softmax(cam_b,dim=0)
    cam_b = cam_b[1]

    patch_num = 1
    patch_size = 512
    if json_data is not None:
        for _pos in range(len(json_data['positive'])):
            _,_,_,_,area = get_area(np.array(json_data['positive'][_pos]['vertices']),width_height=[width,height])
            if area >= (patch_size*patch_num)**2:
                print(_pos,area**0.5 / patch_size)



    if roi_coord is None:
        if roi >= 0:
            if json_data is not None:
                pos_anchors = np.array(json_data['positive'][roi]['vertices'])
                top,down,left,right,_ = get_area(pos_anchors,margin_percentage,width_height=[width,height])
            else:
                # normal slide 我找的区域是ab-mil attention最高的那个图块的4周，_a_idx[0]就是这个。可以更换其他点，比如换一下0，或者换成cam排序最高的，_cam_idx
                top,down,left,right,_ = get_area(None,margin_percentage,coords[_a_idx[0]],width_height=[width,height])
        else:
            top,down,left,right = 0,height,0,width
        crop_coords = []
        # print(top,down,left,right)
        for i in range(top, down, stride):
            for j in range(left, right, stride):
                if j + crop_size > width or i + crop_size > height:
                    continue
                crop_coords.append((j, i))
        right, down = np.max(crop_coords, 0) + crop_size
    else:
        top,down,left,right = roi_coord
    print(">>>>>>>>>>>> region coord gotten >>>>>>>>>>>")

    scale_level_ratio = np.array(slide.level_dimensions[vis_level]) / np.array(slide.level_dimensions[0])
    _w,_h = np.array((right-left, down-top))*scale_level_ratio
    # _w,_h = (right-left, down-top)
    region = slide.read_region((left, top), list(range(slide.level_count))[vis_level], (int(_w),int(_h))).convert('RGB')
    print(">>>>>>>>>>>> slide region gotten >>>>>>>>>>>")
    #

    A_roi,coords_roi,cam_roi = screen_coords(A_b,coords,(left,top),(right,down),cam=cam_b)
    print(">>>>>>>>>>cam_roi>>>>>>>>>>>>")
    print(cam_roi)
    #
    img = np.array(region)
    #
    coords_roi_rel = []
    for i in range(len(coords_roi)):
        rel_x = coords_roi[i][0]-left
        rel_y = coords_roi[i][1] - top
        rel_x,rel_y = np.clip(rel_x,0,right-left),np.clip(rel_y,0,down-top)
        coords_roi_rel.append([rel_x,rel_y])
    print(">>>>>>>>>>>> relative coords computed >>>>>>>>>>>")
    #
    # 正则化
    # roi区域内正则化
    if is_norm:
        if rel_norm:
            A_roi = (A_roi - A_roi.min()) / (A_roi.max() - A_roi.min())
        #全局正则化
        else:
            A_roi = (A_roi - A_b.min()) / (A_b.max() - A_b.min())
        # 筛选
    A_roi[A_roi<filter_thr] = 0
    cam_roi[cam_roi <= filter_thr_cam] = 0

    heatmap = np.zeros(img.shape)
    # attention染色
    for _idx,_coord in enumerate(coords_roi_rel):
        # 这里可以改颜色，rgb
        _c = np.array([255,255,255])
        _color = cam_roi[_idx] * _c if cam_roi[_idx] != 0 else A_roi[_idx] * 255
        heatmap[int(_coord[1]*scale_level_ratio[1]):int((_coord[1]+patch_size)*scale_level_ratio[1]),int(_coord[0]*scale_level_ratio[0]):int((_coord[0]+patch_size)*scale_level_ratio[0]),:] = _color

    print(">>>>>>>>>>>> heatmap computed >>>>>>>>>>>")
    #
    blended_image = (alpha * img + (1-alpha) * heatmap[:, :, :3])
    blended_image = blended_image.astype(np.uint8)

    if vis_size is not None:
        plt.imshow(cv2.resize(blended_image,dsize=(vis_size,vis_size)))
        scale_ratio = (vis_size / img.shape[0])*scale_level_ratio
    else:
        plt.imshow(blended_image)
        scale_ratio = scale_level_ratio
    print(">>>>>>>>>>>> heatmap draw done>>>>>>>>>>>")
    plt.xlim(0,(right-left)*scale_ratio[0])
    plt.ylim((down-top)*scale_ratio[1],0)
    plt.axis("off")

    if json_data is not None:
        for anchors in (json_data['positive']):
            _pos_anchors = np.array(anchors['vertices'])
            plt.plot((_pos_anchors[:, 0]-left) * scale_ratio[0] , (_pos_anchors[:,1]-top)* scale_ratio[1], color='deepskyblue',linewidth=1.0)

    path = os.path.join("./img/",f_id+'level-'+str(level)+'anttehion.png')
    plt.savefig(path,dpi=450,bbox_inches='tight')
    plt.close()



# TSNE--------------------------------------------------------------------------------------------------------------------------------------------------------
def demo_tsne_topk(perplexity=30, n_iter=1200,random_state=0,file_id=None,feature=None,attn=None,level=None):
    X_t = feature         # (N, 512)
    attn_t = attn        # (N,)
   
    # 2) to numpy
    N,D =  feature.shape
    top_k =  int(N * 0.1)
    X = X_t.detach().cpu().numpy().astype(np.float32)
    attn = attn_t.detach().cpu().numpy().astype(np.float32)

    # 3) t-SNE -> (N, 2)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        verbose=1,
    )
    Y = tsne.fit_transform(X)

    # 4) top-k by attention
    top_k = min(top_k, N)
    top_idx = np.argpartition(attn, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(attn[top_idx])[::-1]]  # sort desc
    mask_top = np.zeros(N, dtype=bool)
    mask_top[top_idx] = True

    # 5) plot
    plt.figure(figsize=(2, 3))

    # others
    plt.scatter(
        Y[~mask_top, 0], Y[~mask_top, 1],
        s=15, c="gray", alpha=0.45, linewidths=0,
        label=f"Visual Information"
    )
    # top-k
    plt.scatter(
        Y[top_idx, 0], Y[top_idx, 1],
        s=35, c="red", alpha=1, linewidths=0.5, edgecolors="black",
        label=f"{level} Semantic Retrieved"
    )
    plt.axis("off")
    plt.legend(loc='upper right')
    plt.tight_layout()
    path = os.path.join("./tsne/"+str(level)+'/'+str(file_id)+'.png')
    plt.savefig(path,dpi=450,bbox_inches='tight')














