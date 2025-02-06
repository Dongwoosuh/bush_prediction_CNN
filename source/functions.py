
import numpy as np
import torch
from torch import nn
# from network.CNN import build
import torch.nn.init as init
import warnings
warnings.filterwarnings(action='ignore')

__all__ = ["monotonicity_loss", "smoothness_loss", "linear_stiffness_extaraction_each", "calculate_wmape", "initialize_weights"]
def monotonicity_loss(output):
    """
    output이 x와 y 방향으로 단조 증가하도록 유도하는 손실 함수.
    """

    x_diff = output[:, 1:, :] - output[:, :-1, :]  # x 방향 차분
    y_diff = output[:, :, 1:] - output[:, :, :-1]  # y 방향 차분
    
    # x_diff와 y_diff가 음수일 경우 벌칙을 부여
    x_loss = torch.mean(torch.relu(-x_diff))  # 음수인 경우만 손실에 추가
    y_loss = torch.mean(torch.relu(-y_diff))
    
    return  (x_loss) + (0.1*y_loss)

def smoothness_loss(output, weight=1.0):
    """
    부드러운 단조 증가를 유도하는 손실 함수 (2차 미분 패널티 추가)
    """
    if isinstance(output, np.ndarray):
        output = torch.tensor(output, dtype=torch.float32)

    # 1차 미분 (기존 monotonicity_loss와 동일)
    x_diff = output[:, 1:, :] - output[:, :-1, :]
    y_diff = output[:, :, 1:] - output[:, :, :-1]

    # 2차 미분 (변화율의 변화 제한)
    x_diff2 = x_diff[:, 1:, :] - x_diff[:, :-1, :]
    y_diff2 = y_diff[:, :, 1:] - y_diff[:, :, :-1]

    # 패널티 적용
    smoothness = torch.mean(torch.abs(x_diff2)) + torch.mean(torch.abs(y_diff2))

    return weight * smoothness


def linear_stiffness_extaraction_each(df, main_axes_inverse):
    slope_list = []
    slope_list2 = []
    slope_gap_list = []

    ## change (N/mm, N-mm/deg) to (kgf/mm, kgf-cm/deg)
    df_x = np.linspace(0, main_axes_inverse, len(df))
    for j in range(int(len(df))):
        if j == 0 or j == len(df) -1:
            slope_list.append(0)
            slope_list2.append(0)
            slope_gap_list.append(0)
        else :
            slope = df[j] / df_x[j] # axis 값과 Force_or_Moment 값의 비율 계산 (데이터 위치에 따라 달라질 수 있음)
            slope2 = (df[j+1]- df[j-1]) / (df_x[j+1]- df_x[j-1])
            slope_gap = slope2/slope
            slope_list.append(slope)
            slope_list2.append(slope2)
            slope_gap_list.append(slope_gap)

    # slope_gap_list에서 1에 가장 가까운 값의 index를 찾아 해당 index의 변환된 slope_list 값을 반환
    if slope_gap_list:
        min_gap_index = min(range(len(slope_gap_list)), key=lambda i: abs(slope_gap_list[i] - 1))
        linear_stiffness = slope_list[min_gap_index]
    else:
        linear_stiffness = None

    return linear_stiffness


def calculate_wmape(predictions, targets):

    wmape = (np.sum(np.abs(targets - predictions)) / np.sum(np.abs(targets))) * 100
    acc4 = 100 - wmape
    acc4 = np.where((acc4>100.), 0., acc4)
    acc4 = np.where((acc4<0.), 0., acc4)
    return acc4

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)