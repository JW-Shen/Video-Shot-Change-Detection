import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision import transforms, models

from typing import List, Union, Dict, Any, Optional, Tuple

class ResNet18(nn.Module):
    """Create a resnet18 model to extract features."""
    def __init__(self):
        super(ResNet18, self).__init__()

        resnet18 = models.resnet18(weights="DEFAULT")
        self.features = nn.Sequential(*list(resnet18.children())[:-1])

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters:
            x: input image
        
        Return:
            output: extracted features
        
        Shape:
            source: (B, C, H, W)
            output: (D)
        """
        x = self.features(x)
        output = x.flatten()
        
        return output
    
def read_ground(vedioname: str) -> List[int]:
    """
    Read groung truth file.

    Parameters:
        vedioname: vedio name
        
    Return:
        grounds: list of ground truth
    """
    grounds = []

    with open(f"./data/{vedioname}_ground.txt", "r") as file:
        lines = file.readlines()
        for line in lines[4:]:
            if "~" in line:
                tmp = line.strip().split("~")
                grounds.append(range(int(tmp[0]), int(tmp[1])+1))
            else:
                grounds.append(int(line.strip()))
    
    return grounds

def evaluation(
    ground: List[int], 
    predict: List[int]
) -> Tuple[float, float, float]:
    """
    Run evaluation process.

    Parameters:
        ground: ground trurh
        predict: prediction

    Return:
        precision: Precision
        recall: Recall
        f1_score: F1 score
    """
    for i, j in enumerate(ground):
        if isinstance(j, range):
            for k in list(j):
                if k in predict:
                    ground[i] = k
                    break
        if isinstance(ground[i], range):
            ground[i] = list(j)[0]

    tp = 0
    fp = 0
    fn = 0
    for pred in predict:
        if pred in ground:
            tp += 1
        else:
            fp += 1
    for g in ground:
        if g not in predict:
            fn += 1

    safe_div = lambda x, y: 0 if y == 0 else x / y
    precision = safe_div(tp, (tp + fp))
    recall = safe_div(tp, (tp + fn))
    f1_score = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1_score
            

def calculate_histogram(
    frame: np.array
) -> Tuple[np.array, np.array, np.array]:
    """
    Calculate histogram.

    Parameters:
        frame: a frame in a video

    Return:
        r_hist: histogram of channel R
        g_hist: histogram of channel G
        b_hist: histogram of channel B
    """
    b, g, r = cv2.split(frame)

    r_hist = cv2.calcHist(images=[r], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    g_hist = cv2.calcHist(images=[g], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    b_hist = cv2.calcHist(images=[b], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

    return r_hist, g_hist, b_hist

def histogram_comparison(
    prev_frame: np.array,
    curr_frame: np.array
) -> float:
    """
    Color histogram comparision.

    Parameters:
        prev_frame: previous frame
        curr_frame: current frame

    Return:
        hist_corr: histogram correaltion between two frame
    """
    r_hist_prev, g_hist_prev, b_hist_prev = calculate_histogram(prev_frame)
    r_hist_curr, g_hist_curr, b_hist_curr = calculate_histogram(curr_frame)

    hist_corr = (
        (
            cv2.compareHist(r_hist_prev, r_hist_curr, cv2.HISTCMP_CORREL) +
            cv2.compareHist(g_hist_prev, g_hist_curr, cv2.HISTCMP_CORREL) +
            cv2.compareHist(b_hist_prev, b_hist_curr, cv2.HISTCMP_CORREL)
        ) / 3
    )

    return hist_corr

def ECR(
    prev_frame: np.array,
    curr_frame: np.array
) -> float:
    """
    Edge Change Ratio.

    Parameters:
        prev_frame: previous frame
        curr_frame: current frame

    Return:
        ecr: edge change ratio
    """
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_edge = cv2.Canny(prev_frame, 50, 200)
    curr_edge = cv2.Canny(curr_frame, 50, 200)
    prev_inv = (255 - prev_edge)
    curr_inv = (255 - curr_edge)
    exit_edge = np.sum(prev_edge & curr_inv)
    enter_edge = np.sum(curr_edge & prev_inv)
    prev_edge_pixel_num = np.sum(prev_edge)
    curr_edge_pixel_num = np.sum(curr_edge)

    safe_div = lambda x, y: 0 if y == 0 else x / y
    ecr = max(safe_div(enter_edge, curr_edge_pixel_num), safe_div(exit_edge, prev_edge_pixel_num))

    return ecr

def motion_vectors(
    prev_frame: np.array,
    curr_frame: np.array
) -> float:
    """
    Motion Vectors.

    Parameters:
        prev_frame: previous frame
        curr_frame: current frame

    Return:
        l1_dis: L1 distance of motion vectors between two frames
    """
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_frame,
        next=curr_frame,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    h, w = prev_frame.shape
    l1_dis = np.sum(abs(flow[...,0] - flow[...,1])) / (h * w)

    return l1_dis

def twin_comparison(
    prev_frame: np.array,
    curr_frame: np.array,
    Tb: float,
    Ts: float,
    accumulate: float = None
) -> Tuple[bool, Union[float, None]]:
    """
    Twin-Comparison Approach.

    Parameters:
        prev_frame: previous frame
        curr_frame: current frame
        Tb: high threshold
        Ts: low threshold
        accumulate: accumulated difference
    """
    hist_diff = 1 - histogram_comparison(prev_frame, curr_frame)
    if hist_diff > Tb:
        return True, None
    elif hist_diff > Ts:
        if accumulate is None:
            return False, hist_diff
        else:
            accumulate += hist_diff
            if accumulate > Tb:
                return True, None
            else:
                return False, accumulate
    else:
        return False, None
    
def cnn(
    model: nn.Module, 
    prev_frame: np.array,
    curr_frame: np.array
) -> float:
    """
    CNN (ResNet18).

    Parameters:
        model: CNN model
        prev_frame: previous frame
        curr_frame: current frame

    Return:
        l1_dis: L1 distance of extracted features between two frames
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    prev_frame = transform(prev_frame).unsqueeze(0)
    curr_frame = transform(curr_frame).unsqueeze(0)

    with torch.no_grad():
        prev_feature = model(prev_frame)
        curr_feature = model(curr_frame)

    l1_dis = F.l1_loss(prev_feature, curr_feature).item()

    return l1_dis

def post_process(shot_cahnge: List[int], filter: int = 30) -> List[int]:
    """
    Post-process.

    Parameters:
        shot_change: shot change prediction
        filter: used to determine how close consecutive 
                two values should be filtered out

    Return:
        filtered: prediction after filtering
    """
    filtered = [shot_cahnge[0]]

    for i in shot_cahnge[1:]:
        if abs(i - filtered[-1]) > filter:
            filtered.append(i)

    return filtered

def shot_change_detection(
    videoname: str, 
    algorithm: str = "histogram", 
    threshold: float = 0.5
) -> List[int]:
    """
    Shot change detection.

    Parameters:
        videoname: video name
        algorithm: algorithm for shot change detection
        threshold: threshold to determine whether the shot change occurs

    Return:
        shot_change: list of frame number at which the shot change occurs 
    """
    print(f"##### {videoname} - {algorithm} #####")

    file_map = {
        "climate": "climate.mp4",
        "news": "news.mpg",
        "ngc": "ngc.mpeg",
    }
    idx_map = {
        "climate": 1780,
        "news": 1379,
        "ngc": 1059,
    }
    
    cap = cv2.VideoCapture(f"./data/{file_map[videoname]}")
    if not cap.isOpened():
        print("Error: The video file cannot be opened.")
        return
    
    shot_cahnge = []
    accumulate = None
    idx = 2 if videoname == "climate" else 1
    model = ResNet18() if algorithm == "cnn" else None

    ret, prev_frame = cap.read()

    while True:
        ret, curr_frame = cap.read()
        if (not ret) or (idx > idx_map[videoname]): break

        if algorithm == "histogram":
            hist_corr = histogram_comparison(prev_frame, curr_frame)
            if hist_corr < threshold:
                shot_cahnge.append(idx)
        elif algorithm == "ECR":
            ecr = ECR(prev_frame, curr_frame)
            if ecr > threshold:
                shot_cahnge.append(idx)
        elif algorithm == "motion":
            l1_dis = motion_vectors(prev_frame, curr_frame)
            if l1_dis > threshold:
                shot_cahnge.append(idx)
        elif algorithm == "twin":
            change, accumulate = twin_comparison(prev_frame, curr_frame, threshold, threshold/5, accumulate)
            if change:
                shot_cahnge.append(idx)
        elif algorithm == "cnn":
            l1_dis = cnn(model, prev_frame, curr_frame)
            if l1_dis > threshold:
                shot_cahnge.append(idx)
            # print(idx, l1_dis)
            
        prev_frame = curr_frame
        idx += 1

    cap.release()
    
    if len(shot_cahnge) >= 2:
        if videoname == "ngc":
            shot_cahnge = post_process(shot_cahnge, 10)
        else:
            shot_cahnge = post_process(shot_cahnge)

    precision, recall, f1_score = evaluation(read_ground(videoname), shot_cahnge)
    
    print(f"Precison: {round(precision, 3)}")
    print(f"Recall: {round(recall, 3)}")
    print(f"F1 Score: {round(f1_score, 3)}")
    print("-" * 30)

    return shot_cahnge

def main(videoname: str, algorithm: str) -> None:
    """
    main

    Parameters:
        videoname: video name
        algorithm: algorithm for shot change detection
    """
    threshold = {
        "climate": {
            "histogram": 0.8,
            "ECR": 0.5,
            "motion": 2.3,
            "twin": 0.2,
            "cnn": 0.03,
        },
        "news": {
            "histogram": 0.8,
            "ECR": 0.3,
            "motion": 3.3,
            "twin": 0.2,
            "cnn": 0.041,
        },
        "ngc": {
            "histogram": 0.85,
            "ECR": 0.90,
            "motion": 3.2,
            "twin": 0.15,
            "cnn": 0.043,
        },
    }

    video_list = ["climate", "news", "ngc"]
    algorithm_list = ["histogram", "ECR", "motion", "twin", "cnn"]

    assert videoname in (video_list+["all"]), "Error: unknown video name."
    assert algorithm in (algorithm_list+["all"]), "Error: unsupported algorithm."

    if videoname == "all":
        for video in video_list:
            if algorithm == "all":
                for alg in algorithm_list:
                    shot_cahnge = shot_change_detection(video, algorithm=alg, threshold=threshold[video][alg])
            else:
                shot_cahnge = shot_change_detection(video, algorithm=algorithm, threshold=threshold[video][algorithm])
    elif algorithm == "all":
        for alg in algorithm_list:
            shot_cahnge = shot_change_detection(videoname, algorithm=alg, threshold=threshold[videoname][alg])
    else:
        shot_cahnge = shot_change_detection(videoname, algorithm=algorithm, threshold=threshold[videoname][algorithm])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-name', type=str, default="all")
    parser.add_argument('--algorithm', type=str, default="all")
    args = parser.parse_args()

    main(args.video_name, args.algorithm)