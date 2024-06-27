import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import  matplotlib.animation as animation
width = 600
height = 400

# 영상 처리 threshold
threshold = 50

def load_vid(path, ms = 500, show=True) : # 영상 파일 로드하여 nparray로 반환 (영상 크기 축소 동시 수행)
    width = 600
    height = 400
    
    # 영상 처리 threshold
    threshold = 50
    capture = cv2.VideoCapture(path)

    frames = []
    while capture.isOpened() :
        run, frame = capture.read()

        if run :
            frame = cv2.resize(frame, (width, height))
            if show==True:
                cv2.imshow('video', frame)
                cv2.waitKey(ms)
        else :
            break
        frames.append(frame)
    capture.release()
    cv2.destroyAllWindows()

    return np.array(frames, dtype = 'uint8')

def show_video(video, ms = 500, save=False, file_name='output', show=True) : # 영상 보기
    import matplotlib.pyplot as plt
    import numpy as np
    
    if save:
        height, width, layers = video[0].shape
        size = (width, height)
        fps = 1000/ms
        fourcc = cv2.VideoWriter_fourcc(*'X264')  # You can use other codecs like 'X264', 'MJPG', etc.
        out = cv2.VideoWriter(f'{file_name}.mp4', fourcc, fps, size)

    for i in range(len(video)) :
        
        if save:
            out.write(video[i])  # Write the frame to the video file
        if show==True:
            cv2.imshow('video', video[i])
            cv2.waitKey(ms)
    cv2.destroyAllWindows()
    
    if save:
        out.release()  # Release the VideoWriter object

def show_img(image, title="image"): #이미지 보기
    # cv2.imshow(title, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    plt.figure(figsize=(5, 5))

    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()
    
def show_two_images(image1, image2, title="image"): #이미지 보기
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    
    ax1.imshow(image1, cmap='gray')
    ax1.set_title('original')
    ax1.set_axis_off()
    ax2.imshow(image2, cmap='gray')
    ax2.set_title('converted')
    ax2.set_axis_off()

def show_three_images(image1, image2, image3, title="image"): #이미지 보기
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(14, 4), sharex=True, sharey=True)
    
    ax1.imshow(image1, cmap='gray')
    ax1.set_title('original')
    ax1.set_axis_off()
    ax2.imshow(image2, cmap='gray')
    ax2.set_title('converted 1')
    ax2.set_axis_off()
    ax3.imshow(image3, cmap='gray')
    ax3.set_title('converted 2')
    ax3.set_axis_off()

def to_hsv(video) : # bgr -> hsv로 변환한 영상 반환
    frames = []

    for i in range(len(video)) :
        frame = video[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        frames.append(frame)

    return np.array(frames, dtype = 'uint8')

def to_bgr(video) : # hsv -> bgr로 변환한 영상 반환
    frames = []

    for i in range(len(video)) :
        frame = video[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

        frames.append(frame)

    return np.array(frames, dtype = 'uint8')

from datetime import datetime
def concat_2vid(video1, video2, save=False, file_name=None, show=True) : # 영상 가로로 이어서 보여줌
    if file_name is None:
        file_name = "output_"+datetime.now().strftime("%y%m%d_%H%M")
    video = np.concatenate((video1, video2), axis = 2)
    show_video(video, save=save, file_name=file_name, show=show)

def concat_3vid(video1, video2, video3, save=False,file_name=None, show=True) : # 영상 가로로 이어서 보여줌
    if file_name is None:
        file_name = "output_"+datetime.now().strftime("%y%m%d_%H%M")
    video = np.concatenate((video1, video2, video3), axis = 2)
    show_video(video, save=save, file_name=file_name, show=show)

def concat_4vid(video1, video2, video3, video4, save=False,file_name=None, show=True) : # 영상 가로로 이어서 보여줌
    if file_name is None:
        file_name = "output_"+datetime.now().strftime("%y%m%d_%H%M")
    video = np.concatenate((video1, video2, video3, video4), axis = 2)
    show_video(video, save=save, file_name=file_name, show=show)

def preprocessing(test_file, mask_road = "mask_road2", mask_running="mask_running2", show=True):
    # 기본 영상 크기 (이 사이즈에 맞게 축소됨)
    width = 600
    height = 400
    
    # 영상 처리 threshold
    threshold = 50
    
    # 경로의 영상 로드
    path = f'./Team Project. Bike Video/{test_file}.mp4'
    
    vid_origin = load_vid(path,show=show)
    
    ######원본 영상 hsv로 전환.

    vid_hsv = to_hsv(vid_origin)

    vid_processed = vid_hsv.copy()
    vid_bgr = to_bgr(vid_processed)

    # 도로색 제외에 참고할 마스크 로드
    
    mask_road = cv2.imread(f'./Team Project. Bike Video/{mask_road}.png', cv2.IMREAD_COLOR)
    
    if mask_road is None:
        print('Wrong path:', path)
    mask_road = cv2.resize(mask_road, (width, height))

    ####### 원본 영상에 대한 hue/intensity 히스토그램 생성

    histograms_hue = [0] * len(vid_processed)
    histograms_intensity = [0] * len(vid_processed)
    
    
    for f in range(len(vid_processed)) :
        hue = [0] * 256
        intensity = [0] * 256
    
        for i in range(height) :
            for j in range(width) :
                if mask_road[i, j, 0] == 255 : # mask 영역에 해당되면
                    hue[vid_processed[f, i, j, 0]] += 1
                    intensity[vid_processed[f, i, j, 2]] += 1
    
        histograms_hue[f] = hue
        histograms_intensity[f] = intensity
    
    # output
    # histograms_hue - frame 별 hue histogram
    # histograms_intensity - frame 별 intensity histogram

    # histogram 보정

    for f in range(len(vid_processed)) :
    
        # 원본 histogram
        hue_origin = histograms_hue[f].copy()
        intensity_origin = histograms_intensity[f].copy()
    
        # 수정된 histogram
        hue = histograms_hue[f]
        intensity = histograms_intensity[f]
    
    
        for i in range(256) :
            if (hue_origin[i] < threshold) and (np.sum(hue_origin[max(0, i-10):min(i+11, 256)]) != 0) :
                hue[i] += np.mean(hue_origin[max(0, i-10):min(i+11, 256)])
            if (intensity_origin[i] < threshold) and (np.sum(intensity_origin[max(0, i-10):min(i+11, 256)]) != 0) :
                intensity[i] += np.mean(intensity_origin[max(0, i-10):min(i+11, 256)])
    
        histograms_hue[f] = hue
        histograms_intensity[f] = intensity


    # 장애물/도로 구분해서 표시할 영상 토대 마련

    vid_obstacles = vid_origin.copy()

    # 히스토그램 분포를 토대로 도로/obstacle 구분하기

    for f in range(len(vid_hsv)) :
        hue = histograms_hue[f]
        intensity = histograms_intensity[f]
    
        for i in range(height) :
            for j in range(width) :
    
                if hue[vid_hsv[f, i, j, 0]] < threshold or intensity[vid_hsv[f, i, j, 2]] < threshold :
                    vid_obstacles[f, i, j] = [255, 255, 255]
                else :
                    vid_obstacles[f, i, j] = [0, 0, 0]
    
                # 그림자 제거
                #if vid_hsv[f, i, j, 2] < 30 :
                    #vid_obstacles[f, i, j] = [0, 0, 0]
    
    # output - vid_obstacles (장애물 = 하얀색, 도로 - 검은색으로 표기된 영상)

    # 주행 영역 내 장애물만 남기기 위한 마스크 로드
    
    mask_running = cv2.imread(f'./Team Project. Bike Video/{mask_running}.png', cv2.IMREAD_COLOR)
    mask_running = cv2.resize(mask_running, (width, height))

    # show_img(mask_running)

    ###### 주행 영역 내 장애물만 남겨 표시할 영상 토대 마련

    vid_running = vid_obstacles.copy()

    # 마스크를 토대로 주행 영역 내 장애물만 남김

    for f in range(len(vid_obstacles)) :
    
        for i in range(height) :
            for j in range(width) :
    
                if mask_running[i, j, 0] == 255 : # mask 영역에 해당되면
                    vid_running[f, i, j] = vid_obstacles[f, i, j]
                else : # mask 영역 밖이면
                    vid_running[f, i, j] = [0, 0, 0]
    
    # output - vid_running (주행 영역 내의 장애물만 남긴 영상)
    
    # 원본 영상 - 장애물 binary 영상 - 주행 영역만 남긴 영상 연결해서 볼 수 있음.

    # concat_3vid(vid_origin, vid_obstacles, vid_running)

    return vid_origin, vid_obstacles, vid_running

