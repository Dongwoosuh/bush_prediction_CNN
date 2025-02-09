import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

import pytorch_model_summary

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, BN_momentum=0.2):
        """
        인자:
          - in_channels: 입력 채널 수
          - out_channels: 출력 채널 수
          - upsample: True인 경우 stride=2를 사용하여 공간 해상도를 두 배로 증가시킴
          - BN_momentum: 배치 정규화의 momentum 값
        """
        super(ResidualBlock, self).__init__()
        self.upsample = upsample
        if upsample:
            # 업샘플링 시 output_padding=1을 통해 정확히 두 배 해상도 출력
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                            kernel_size=3, stride=2, padding=1, output_padding=1)
        else:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                            kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)
        self.activation = nn.SiLU()
        # 두 번째 convolution은 해상도 변화 없이 특성 추출
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                        kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_momentum)

        # shortcut: 입력과 출력의 채널 수 또는 해상도가 다르면 1×1 transposed convolution으로 보정
        if upsample or in_channels != out_channels:
            if upsample:
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels,
                                       kernel_size=1, stride=2, padding=0, output_padding=1),
                    nn.BatchNorm2d(out_channels, momentum=BN_momentum)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels,
                                       kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(out_channels, momentum=BN_momentum)
                )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.activation(out)
        return out

class build(nn.Module):
    def __init__(self, num_DV):
        super(build, self).__init__()
        BN_momentum = 0.2

        self.start_ch = 1024  # 초기 채널 수
        # 완전 연결층: 입력 벡터를 (start_ch x 2 x 2) 텐서로 변환
        self.fc = nn.Sequential(
            nn.Linear(in_features=num_DV, out_features=self.start_ch * 2 * 2),
            nn.BatchNorm1d(self.start_ch * 2 * 2, momentum=BN_momentum),
            nn.SiLU()
        )

        # fc 출력(1024, 2, 2)을 시작으로 잔차 블록을 통해 점진적으로 채널 수 감소 및 업샘플링 진행
        self.resblock1 = ResidualBlock(self.start_ch, self.start_ch // 2, upsample=True, BN_momentum=BN_momentum)   # (1024,2,2) -> (512,2,2)
        # self.resblock2 = ResidualBlock(self.start_ch // 2, self.start_ch // 2, upsample=True, BN_momentum=BN_momentum)  # (512,2,2) -> (512,4,4)
        self.resblock3 = ResidualBlock(self.start_ch // 2, self.start_ch // 4, upsample=True, BN_momentum=BN_momentum)  # (512,4,4) -> (256,8,8)
        self.resblock4 = ResidualBlock(self.start_ch // 4, self.start_ch // 8, upsample=True, BN_momentum=BN_momentum)  # (256,8,8) -> (128,16,16)
        self.resblock5 = ResidualBlock(self.start_ch // 8, self.start_ch // 16, upsample=False, BN_momentum=BN_momentum) # (128,16,16) -> (64,16,16)

        # 출력층: 채널 수를 1로 축소하고 Sigmoid를 적용하여 최종 결과 도출
        self.conv_last = nn.Sequential(
            nn.ConvTranspose2d(self.start_ch // 16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        # 완전 연결층 통과 후 (batch, 1024, 2, 2) 텐서로 재구성
        x = self.fc(input)
        x = x.view(-1, self.start_ch, 2, 2)
        # 잔차 블록을 통한 특성 추출 및 업샘플링
        x = self.resblock1(x)  # -> (512, 2, 2)
        # x = self.resblock2(x)  # -> (512, 4, 4)
        x = self.resblock3(x)  # -> (256, 8, 8)
        x = self.resblock4(x)  # -> (128, 16, 16)
        x = self.resblock5(x)  # -> (64, 16, 16)
        x = self.conv_last(x)  # -> (1, 16, 16)
        # 최종 출력: (batch, 16, 16)
        x = x.view(-1, 16, 16)
        return x

if __name__ == "__main__": 
    model = build(num_DV=12)
    input_tensor = torch.zeros(1, 12)  # (batch, in_features)
    model_summary = pytorch_model_summary.summary(model, input_tensor, show_input=True)
    print(model_summary)
