import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

import pytorch_model_summary

class build(nn.Module):

    def __init__(self,num_DV):
        super(build, self).__init__()
        BN_momentum = 0.2
        dropout_rate = 0.2

        self.start_ch = 1024  # 초기 채널 수 수정
        self.padding_param = 1
        self.kernel_size = 4
        self.stride = 2
        # 완전 연결층
        # self.fc = nn.Sequential(
            # nn.Linear(in_features=num_DV, out_features=self.start_ch * 2 * 2),        # (batch_size, 2048)
            # nn.LeakyReLU(),
            # )
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=num_DV, out_features=self.start_ch),        # (batch_size, 2048)
            nn.SiLU(),
            nn.Linear(in_features=self.start_ch, out_features=self.start_ch * 2 *2),        # (batch_size, 2048)
            nn.SiLU(),
            # nn.Linear(in_features=self.start_ch * 2, out_features=self.start_ch * 2 * 2),        # (batch_size, 2048)
            # nn.SiLU()
            )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(self.start_ch, self.start_ch // 4, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding_param),  # (batch_size, 256, 4, 4)
            nn.BatchNorm2d(self.start_ch // 4, momentum=BN_momentum),
            nn.SiLU(),
            nn.Dropout(dropout_rate),

            nn.ConvTranspose2d(self.start_ch // 4, self.start_ch // 8, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding_param),  # (batch_size, 256, 6, 6)
            nn.BatchNorm2d(self.start_ch // 8, momentum=BN_momentum),
            nn.SiLU(),
            nn.Dropout(dropout_rate),

            # nn.ConvTranspose2d(self.start_ch // 8, self.start_ch // 16, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding_param),  # (batch_size, 128, 13, 13)
            # nn.BatchNorm2d(self.start_ch // 16, momentum=BN_momentum),
            # nn.SiLU(),
            # nn.ConvTranspose2d(self.start_ch // 8, self.start_ch // 16, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding_param),  # (batch_size, 128, 28, 28)
            # nn.BatchNorm2d(self.start_ch // 16, momentum=BN_momentum),
            # nn.SiLU(),
            # nn.Dropout(dropout_rate)
        )

        # 출력층 수정
        self.conv_last = nn.Sequential(
            nn.ConvTranspose2d(self.start_ch // 8, 1, kernel_size=4, stride=2, padding=1),  # (batch_size, 1, 32, 32)
            nn.Sigmoid()
        ) # Sigmoid

    def forward(self, input):
        # 완전 연결층 통과 후 재구성
        x = self.fc(input)  # (batch_size, 2048)
        x = x.view(-1, self.start_ch, 2, 2)  # (batch_size, 512, 2, 2)
        x = self.conv5(x)  # (batch_size, 128, 28, 28)
        # x = self.conv_last(x)
        x = self.conv_last(x).view([-1,16,16])
        return x
    
if __name__ == "__main__": 
    model = build(num_DV=12)
    input_tensor = torch.zeros(1, 12)  # (batch_size, in_features)
    model_summary = pytorch_model_summary.summary(model, input_tensor, show_input=True)
    print(model_summary)
