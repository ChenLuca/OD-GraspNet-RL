import torch.nn as nn
import torch.nn.functional as F
import torch
from inference.models.RJ_grasp_model import GraspModel, OSAModule, OSABlock, TransitionBlock
from inference.models.rfb import BasicRFB_a

class Generative_ODR_1_CSP(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=256, dropout=False, prob=0.0):
        super(Generative_ODR_1_CSP, self).__init__()

        print("Generative_ODR_1_CSP")
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.conv_out_size = channel_size * 4

        #osa-dense
        self.osa_depth = 5
        self.osa_conv_kernal = [32, 40, 48, 56]
        self.trans_conv_kernal = [64, 128, 192, 256]
        self.osa_drop_rate = 0.0
        self.osa_reduction = 1.0
        
        #csp
        self.csp_part_ratio = 0.5

        self.part_1_channels_1 = int(self.conv_out_size*self.csp_part_ratio)
        self.part_2_channels_1 = self.conv_out_size - self.part_1_channels_1


        self.csp_trans1_out_kernal = self.trans_conv_kernal[0] * 2 


        self.part_1_channels_2 = int(self.csp_trans1_out_kernal*self.csp_part_ratio)
        self.part_2_channels_2 = self.csp_trans1_out_kernal - self.part_1_channels_2

        self.csp_trans2_out_kernal = self.trans_conv_kernal[1] * 2


        self.part_1_channels_3 = int(self.csp_trans2_out_kernal*self.csp_part_ratio)
        self.part_2_channels_3 = self.csp_trans2_out_kernal - self.part_1_channels_3

        self.csp_trans3_out_kernal = self.trans_conv_kernal[2] *2

        self.part_1_channels_4 = int(self.csp_trans3_out_kernal*self.csp_part_ratio)
        self.part_2_channels_4 = self.csp_trans3_out_kernal - self.part_1_channels_4

        self.csp_trans4_out_kernal = self.trans_conv_kernal[3] *2


        # 1st block
        self.block1 = OSABlock(self.osa_depth, self.part_2_channels_1, self.osa_conv_kernal[0], OSAModule, self.osa_drop_rate)
        self.osa_trans1 = TransitionBlock(self.osa_conv_kernal[0]*self.osa_depth, self.trans_conv_kernal[0], dropRate=self.osa_drop_rate)
        self.csp_trans1 = TransitionBlock(self.part_1_channels_1, self.trans_conv_kernal[0], dropRate=self.osa_drop_rate)

        
        # 2nd block
        self.block2 = OSABlock(self.osa_depth, self.part_2_channels_2, self.osa_conv_kernal[1], OSAModule, self.osa_drop_rate)
        self.osa_trans2 = TransitionBlock(self.osa_conv_kernal[1]*self.osa_depth, self.trans_conv_kernal[1], dropRate=self.osa_drop_rate)
        self.csp_trans2 = TransitionBlock(self.part_1_channels_2, self.trans_conv_kernal[1], dropRate=self.osa_drop_rate)


        # 3rd block
        self.block3 = OSABlock(self.osa_depth, self.part_2_channels_3, self.osa_conv_kernal[2], OSAModule, self.osa_drop_rate)
        self.osa_trans3 = TransitionBlock(self.osa_conv_kernal[2]*self.osa_depth, self.trans_conv_kernal[2], dropRate=self.osa_drop_rate)
        self.csp_trans3 = TransitionBlock(self.part_1_channels_3, self.trans_conv_kernal[2], dropRate=self.osa_drop_rate)


        # 4rd block
        self.block4 = OSABlock(self.osa_depth, self.part_2_channels_4, self.osa_conv_kernal[3], OSAModule, self.osa_drop_rate)
        self.osa_trans4 = TransitionBlock(self.osa_conv_kernal[3]*self.osa_depth, self.trans_conv_kernal[3], dropRate=self.osa_drop_rate)
        self.csp_trans4 = TransitionBlock(self.part_1_channels_4, self.trans_conv_kernal[3], dropRate=self.osa_drop_rate)

        self.rfb = BasicRFB_a(self.csp_trans4_out_kernal, self.csp_trans4_out_kernal)

        self.conv4 = nn.ConvTranspose2d(self.csp_trans4_out_kernal, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)

        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)

        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):

        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # print("conv_x.shape ", x.shape)

        part_1 = x[:,:self.part_1_channels_1, :, :]
        part_2 = x[:,self.part_1_channels_1:, :, :]
        x_part_1 = self.csp_trans1(part_1)
        x_part_2 = self.osa_trans1(self.block1(part_2))
        x = torch.cat((x_part_1, x_part_2), 1)
        # print("part_1.shape ", part_1.shape)
        # print("part_2.shape ", part_2.shape)
        # print("cat_1.shape ", x.shape)

        part_1 = x[:,:self.part_1_channels_2, :, :]
        part_2 = x[:,self.part_1_channels_2:, :, :]
        x_part_1 = self.csp_trans2(part_1)
        x_part_2 = self.osa_trans2(self.block2(part_2))
        x = torch.cat((x_part_1, x_part_2), 1)
        # print("part_1.shape ", part_1.shape)
        # print("part_2.shape ", part_2.shape)
        # print("cat_2.shape ", x.shape)

        part_1 = x[:,:self.part_1_channels_3, :, :]
        part_2 = x[:,self.part_1_channels_3:, :, :]
        x_part_1 = self.csp_trans3(part_1)
        x_part_2 = self.osa_trans3(self.block3(part_2))
        x = torch.cat((x_part_1, x_part_2), 1)
        # print("part_1.shape ", part_1.shape)
        # print("part_2.shape ", part_2.shape)
        # print("cat_3.shape ", x.shape)

        part_1 = x[:,:self.part_1_channels_4, :, :]
        part_2 = x[:,self.part_1_channels_4:, :, :]
        x_part_1 = self.csp_trans4(part_1)
        x_part_2 = self.osa_trans4(self.block4(part_2))
        x = torch.cat((x_part_1, x_part_2), 1)
        # print("part_1.shape ", part_1.shape)
        # print("part_2.shape ", part_2.shape)
        # print("cat_4.shape ", x.shape)

        x = self.rfb(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output
