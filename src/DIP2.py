from DIP import *

# opening, closing 진행한 후 결과 확인
def morphological_filtering(video):
    from skimage.morphology import erosion, dilation, opening, closing, white_tophat  # noqa
    from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
    from skimage.morphology import disk 
    footprint = np.stack([disk(3)]*3, axis=0)
    # footprint_2 = np.stack([disk(5)]*3, axis=0)
    
    
    vid_opening = [] #vid_running.copy()
    vid_closing = [] #vid_running.copy()
    for frame_n in range(len(video)) :
        opened = opening(video[frame_n], footprint)
        # closed = closing(video[frame_n], footprint)
        closed = closing(opened, footprint)
    
        vid_opening.append(opened)
        vid_closing.append(closed)
    return vid_opening, vid_closing
    
# opening, closing 진행한 후 결과 확인
def morphological_filtering_closetoopen(video):
    from skimage.morphology import erosion, dilation, opening, closing, white_tophat  # noqa
    from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
    from skimage.morphology import disk 
    footprint = np.stack([disk(3)]*3, axis=0)
    # footprint_2 = np.stack([disk(5)]*3, axis=0)
    
    
    vid_opening = [] #vid_running.copy()
    vid_closing = [] #vid_running.copy()
    for frame_n in range(len(video)) :
        closed = closing(video[frame_n], footprint)
        # closed = closing(video[frame_n], footprint)
        opened = opening(closed, footprint)
    
        vid_closing.append(closed)
        vid_opening.append(opened)

    return vid_closing, vid_opening

# output - vid_opening, vid_closing

import numpy as np
import cv2
from matplotlib import pyplot as plt

def add_to_dict(dict, key, value):
    if key in dict:
        dict[key].append(value)
        dict[key] = list(set(dict[key]))
    else:
        dict[key] = [value]
        
def CCL_image(image,frame_num):
    uf = {}
    if len(image.shape) == 3:
        image = image[:,:,0]
        
    def first_pass(image):
        rows, cols = image.shape
        labeled_image = np.zeros_like(image)
        label_counter = 0
    
        def union(px1, px2):
            root1 = get_label(px1)
            root2 = get_label(px2)
            uf[root2] = root1
    
        for i in range(rows):
            for j in range(cols):
                if image[i, j] != 0:
                    neighbors = [(i-1, j), (i, j-1)]
                    neighbor_labels = [labeled_image[x, y] for x, y in neighbors if 0 <= x < rows and 0 <= y < cols]
                        
                    if neighbor_labels and neighbor_labels.count(0) == len(neighbor_labels):
                        label_counter += 1
                        labeled_image[i, j] = label_counter
                    else:
                        neighbor_labels = [label for label in neighbor_labels if label != 0]
                        labeled_image[i, j] = min(neighbor_labels)
                        for label in neighbor_labels:
                            add_to_dict(uf, labeled_image[i,j], label)
        return labeled_image
    
    def second_pass(image):
        rows, cols = image.shape
        label_l = set([])
        for i in range(rows):
            for j in range(cols):
                px = image[i, j]
                if px != 0:
                    min_label = min([k for k, v in uf.items() if px in v])
                    label_l.update([min_label])
                    image[i, j] = min_label
        if len(np.unique(label_l))!=1:
            print("Image num:",frame_num,"Number of labels:", len(np.unique(label_l)) )  # Exclude background label

        return image

    first_image = first_pass(image)

    uf_pass = {k:[k] for k in range(1,len(np.unique(first_image)) )} #uf.copy()
    for key, value in reversed(sorted(uf.items())):
        # print(key, ":", value)
        if len(value) != 1:
            for v in value:
                if v != key and v in uf.keys():
                    uf_pass[key].extend(uf[v])

    uf = {k:v for k,v in uf_pass.items() }
    second_image = second_pass(first_image)
    # print("Image num:",frame_num,"Number of labels:", len(np.unique(labeled_image)) - 1)  # Exclude background label

    return second_image
    
def connected_component_labeling(video):
    vid_ccl = []
    for frame_num in range(len(video)):
        labeled_image = CCL_image(video[frame_num].astype(np.uint8),frame_num)
        vid_ccl.append(labeled_image)

    vid_ccl_view = np.stack([  [[[200 if pix==0 else pix  for pix in row] for row in img] for img in vid_ccl]  ]*3, axis=3).astype(np.uint8)
    vid_ccl = np.stack([ vid_ccl ]*3, axis=3)

    return vid_ccl, vid_ccl_view

def connected_component_labeling_view(video):
    vid_ccl = []; vid_ccl_view=[]
    for frame_num in range(len(video)):
        labeled_image = CCL_image(video[frame_num].astype(np.uint8),frame_num)
        
        labeld_image_view = np.array([[200 if pix==0 else pix*30  for pix in row] for row in labeled_image]  )
        labeld_image_view = cv2.cvtColor(labeld_image_view.astype(np.uint8), cv2.COLOR_GRAY2BGR)  # 그레이스케일 이미지를 BGR 컬러 이미지로 변환
        for label in np.unique(labeled_image):
            if label == 0 :
                continue
            coords = np.column_stack(np.where(labeled_image == label))
            y_center = coords[:, 0].mean()
            x_center = coords[:, 1].mean()
            label_size = np.sum(labeled_image == label)
            cv2.putText(labeld_image_view, f"label:{label}, size:{label_size}", (int(x_center), int(y_center)),
                       fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 1, color = (255, 255, 255), thickness=2)
            
            
        vid_ccl_view.append(labeld_image_view)
        vid_ccl.append(labeled_image)

    # vid_ccl_view = np.stack([ vid_ccl_view ]*3, axis=3)
    vid_ccl = np.stack([ vid_ccl ]*3, axis=3)
    
    return vid_ccl, vid_ccl_view