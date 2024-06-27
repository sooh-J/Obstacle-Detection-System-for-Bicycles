import numpy as np
import cv2

def segment_and_classify(org_image, binary_image,size_threshold,warn_threshold,threshold):

    if len(np.shape(binary_image)) > 2:
        binary_image  = binary_image[:,:,0]

    # 레이블의 크기를 계산하여 특정 기준치보다 큰 레이블만 분류
    label_sizes = {label: np.sum(binary_image == label) for label in np.unique(binary_image) if label != 0}
    object_labels = {label: size for label, size in label_sizes.items() if (size >= size_threshold) and (size < warn_threshold)}
    warn_labels = {label: size for label, size in label_sizes.items() if size > warn_threshold}

    output_image, output_image_color = draw_boxes(org_image, binary_image, object_labels, warn_labels)
    
    return output_image, output_image_color

def draw_boxes(org_image, binary_image, object_labels, warn_labels):
    output_image_color = org_image.copy()
    output_image = cv2.cvtColor(binary_image*50, cv2.COLOR_GRAY2BGR)
    
    for label, size in object_labels.items():
        coords = np.column_stack(np.where(binary_image == label))
        top_left = (coords[:, 1].min(), coords[:, 0].min())
        bottom_right = (coords[:, 1].max(), coords[:, 0].max())
        
        green = (0, 255, 0)  # Green
        cv2.rectangle(output_image_color, top_left, bottom_right, green, 2)
        cv2.rectangle(output_image, top_left, bottom_right, green, 2)

    for label, size in warn_labels.items():
        coords = np.column_stack(np.where(binary_image == label))
        top_left = (coords[:, 1].min(), coords[:, 0].min())
        bottom_right = (coords[:, 1].max(), coords[:, 0].max())
        
        red = (0, 0, 255)  # Red
        cv2.rectangle(output_image_color, top_left, bottom_right, red, 2)
        cv2.rectangle(output_image, top_left, bottom_right, red, 2)
        
    return output_image, output_image_color
    
def classification_objects(org_video, video, size_threshold=500, warn_threshold=1000, threshold=8):
    output_video = []; output_video_color=[]
    for i in range(len(video)):
        labeled_image, labeled_image_color = \
            segment_and_classify(org_video[i],video[i],size_threshold=size_threshold,warn_threshold=warn_threshold,threshold=threshold)

        output_video.append(labeled_image)
        output_video_color.append(labeled_image_color)

    return org_video, output_video, output_video_color

# output_video, output_video_color = classification_objects(vid_origin.copy(), vid_ccl_open.copy())
# concat_2vid(output_video, output_video_color, save=True)
# concat_2vid(np.stack([ output_video ]*3, axis=3), output_video_color)