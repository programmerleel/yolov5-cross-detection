# yolov5



**代码地址：**

- 代码目录（完成注释）：



### 网络结构

​		**以yolov5l，输入数据尺寸[1,3,640,640]为例进行分析**

- Focus Conv：

  - 作者舍弃了Focus操作 转为替换成为大卷积核的卷积操作，对比下面的输出数据，**虽然使用卷积的运算量和参数两都远远大于Focus，但是实际运行速度却快于Focus**

  - ```
    focus_flops:  104857600.0
    focus_params:  896.0
    focus_time 0.01800084114074707
    conv_flops:  734003200.0
    conv_params:  7040.0
    conv_t 0.014000415802001953
    ```

    

  ![Focus.Conv](assets/Focus.Conv.png)

  



- C3 BottleNeckCSP：

  - 作者设置BottleNeck的通道缩放值为1.0，实际并未达到BottleNeck先降低通道维度的作用

  - BottleNeck存在两种形式，在head中的不存在add

  - **对比C3与BottleNeckCSP**，**精简了网络结构，替换SiLU为激活函数**

  - ```
    csp_flops:  4161536000.0
    csp_params:  161152.0
    csp_time 0.06800007820129395
    c3_flops:  4050124800.0
    c3_params:  156928.0
    csp_t 0.06200146675109863
    ```

    

  ![C3CSP](assets/C3CSP.png)

- SPP SPPF

  - 将SPP替换为SPPF模块，从并行的结构改为串行的结构，参数量与运算量并未发生改变，但是运行的速度加快
  - 融合不同感受野的特征

  - ```
    spp_flops:  1051033600.0
    spp_params:  2624512.0
    spp_time 0.017999649047851562
    sppf_flops:  1051033600.0
    sppf_params:  2624512.0
    sppf_t 0.013000249862670898
    ```

    

  

  ![SPPFSPP](assets/SPPFSPP.png)

- model

  ![model](assets/model.png)

  - [1,256,80,80]

  - [1,512,40,40]

  - [1,1024,20,20]

### 改进思路

- **更改模块**

  - 替换C3模块为C2f模块

    1. models/common.py文件中添加C2f模块

       ```python
       class C2f(nn.Module):
           """Faster Implementation of CSP Bottleneck with 2 convolutions."""
       
           def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
               """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
               expansion.
               """
               super().__init__()
               self.c = int(c2 * e)  # hidden channels
               self.cv1 = Conv(c1, 2 * self.c, 1, 1)
               self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
               self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
       
           def forward(self, x):
               """Forward pass through C2f layer."""
               y = list(self.cv1(x).chunk(2, 1))
               y.extend(m(y[-1]) for m in self.m)
               return self.cv2(torch.cat(y, 1))
       
           def forward_split(self, x):
               """Forward pass using split() instead of chunk()."""
               y = list(self.cv1(x).split((self.c, self.c), 1))
               y.extend(m(y[-1]) for m in self.m)
               return self.cv2(torch.cat(y, 1))
       ```

    2. 修改models/yolov5s.yaml配置文件

       ```yaml
       # YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
       
       # Parameters
       nc: 80 # number of classes
       depth_multiple: 0.33 # model depth multiple
       width_multiple: 0.50 # layer channel multiple
       anchors:
         - [10, 13, 16, 30, 33, 23] # P3/8
         - [30, 61, 62, 45, 59, 119] # P4/16
         - [116, 90, 156, 198, 373, 326] # P5/32
       
       # YOLOv5 v6.0 backbone
       backbone:
         # [from, number, module, args]
         [
           [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2
           [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
           [-1, 3, C2f, [128]],
           [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
           [-1, 6, C2f, [256]],
           [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
           [-1, 9, C2f, [512]],
           [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
           [-1, 3, C2f, [1024]],
           [-1, 1, SPPF, [1024, 5]], # 9
         ]
       
       # YOLOv5 v6.0 head
       head: [
           [-1, 1, Conv, [512, 1, 1]],
           [-1, 1, nn.Upsample, [None, 2, "nearest"]],
           [[-1, 6], 1, Concat, [1]], # cat backbone P4
           [-1, 3, C2f, [512, False]], # 13
       
           [-1, 1, Conv, [256, 1, 1]],
           [-1, 1, nn.Upsample, [None, 2, "nearest"]],
           [[-1, 4], 1, Concat, [1]], # cat backbone P3
           [-1, 3, C2f, [256, False]], # 17 (P3/8-small)
       
           [-1, 1, Conv, [256, 3, 2]],
           [[-1, 14], 1, Concat, [1]], # cat head P4
           [-1, 3, C2f, [512, False]], # 20 (P4/16-medium)
       
           [-1, 1, Conv, [512, 3, 2]],
           [[-1, 10], 1, Concat, [1]], # cat head P5
           [-1, 3, C2f, [1024, False]], # 23 (P5/32-large)
       
           [[17, 20, 23], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)
         ]
       ```

    3. models/yolo.py文件中导入C2f模块

       ```python
       from models.common import C2f
       ```

    4. models/yolo.py中添加代码

       ```python
       # 大概在425行 
       if m in {
                   Conv,
                   GhostConv,
                   Bottleneck,
                   GhostBottleneck,
                   SPP,
                   SPPF,
                   DWConv,
                   MixConv2d,
                   Focus,
                   CrossConv,
                   BottleneckCSP,
                   C3,
                   C3TR,
                   C3SPP,
                   C3Ghost,
                   nn.ConvTranspose2d,
                   DWConvTranspose2d,
                   C3x,
                   C2f
               }:
       ```

  - 添加DCNv2模块

    ***强调（千万注意）：在添加DCN模块时，在BottleNeck模块中将原有的一个Conv替换成为了DCN，组合成为了BottleNeckDCN模块，再以此替换C3中的对应模块，但是在backbone中加入DCN的方式绝对不止这一种，牢记一点，对网络所作的一切修改包括在训练中的某些手段都是为了具体任务而服务的，要根据实际的任务需求去实际测试效果（速度或精度），而不是更改过后就一劳永逸***

    1. models/common.py添加DCN，BottleNecDCN，C3DCN模块

       ```python
       class DCN(nn.Module):
           def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1,bias=None, modulation=False):
               """
               Args:
                   modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
               """
               super(DCN, self).__init__()
               self.kernel_size = kernel_size
               self.padding = padding
               self.stride = stride
               self.zero_padding = nn.ZeroPad2d(padding)
               self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
       
               self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
               nn.init.constant_(self.p_conv.weight, 0)
               self.p_conv.register_backward_hook(self._set_lr)
       
               self.modulation = modulation
               if modulation:
                   self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
                   nn.init.constant_(self.m_conv.weight, 0)
                   self.m_conv.register_backward_hook(self._set_lr)
       
           @staticmethod
           def _set_lr(module, grad_input, grad_output):
               grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
               grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
       
           def forward(self, x):
               offset = self.p_conv(x)
               if self.modulation:
                   m = torch.sigmoid(self.m_conv(x))
       
               dtype = offset.data.type()
               ks = self.kernel_size
               N = offset.size(1) // 2
       
               if self.padding:
                   x = self.zero_padding(x)
       
               # (b, 2N, h, w)
               p = self._get_p(offset, dtype)
       
               # (b, h, w, 2N)
               p = p.contiguous().permute(0, 2, 3, 1)
               q_lt = p.detach().floor()
               q_rb = q_lt + 1
       
               q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
               q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
               q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
               q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
       
               # clip p
               p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)
       
               # bilinear kernel (b, h, w, N)
               g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
               g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
               g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
               g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
       
               # (b, c, h, w, N)
               x_q_lt = self._get_x_q(x, q_lt, N)
               x_q_rb = self._get_x_q(x, q_rb, N)
               x_q_lb = self._get_x_q(x, q_lb, N)
               x_q_rt = self._get_x_q(x, q_rt, N)
       
               # (b, c, h, w, N)
               x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                          g_rb.unsqueeze(dim=1) * x_q_rb + \
                          g_lb.unsqueeze(dim=1) * x_q_lb + \
                          g_rt.unsqueeze(dim=1) * x_q_rt
       
               # modulation
               if self.modulation:
                   m = m.contiguous().permute(0, 2, 3, 1)
                   m = m.unsqueeze(dim=1)
                   m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
                   x_offset *= m
       
               x_offset = self._reshape_x_offset(x_offset, ks)
               out = self.conv(x_offset)
       
               return out
       
           def _get_p_n(self, N, dtype):
               p_n_x, p_n_y = torch.meshgrid(
                   torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
                   torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
               # (2N, 1)
               p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
               p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
       
               return p_n
       
           def _get_p_0(self, h, w, N, dtype):
               p_0_x, p_0_y = torch.meshgrid(
                   torch.arange(1, h*self.stride+1, self.stride),
                   torch.arange(1, w*self.stride+1, self.stride))
               p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
               p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
               p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
       
               return p_0
       
           def _get_p(self, offset, dtype):
               N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
       
               # (1, 2N, 1, 1)
               p_n = self._get_p_n(N, dtype)
               # (1, 2N, h, w)
               p_0 = self._get_p_0(h, w, N, dtype)
               p = p_0 + p_n + offset
               return p
       
           def _get_x_q(self, x, q, N):
               b, h, w, _ = q.size()
               padded_w = x.size(3)
               c = x.size(1)
               # (b, c, h*w)
               x = x.contiguous().view(b, c, -1)
       
               # (b, h, w, N)
               index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
               # (b, c, h*w*N)
               index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
       
               x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
       
               return x_offset
       
           @staticmethod
           def _reshape_x_offset(x_offset, ks):
               b, c, h, w, N = x_offset.size()
               x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
               x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
       
               return x_offset
               
               
       class BottleneckDCN(nn.Module):
           # Standard bottleneck
           def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
               """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
               expansion.
               """
               super().__init__()
               c_ = int(c2 * e)  # hidden channels
               self.cv1 = Conv(c1, c_, 1, 1)
               self.cv2 = DCN(c_, c2, 3, 1)
               self.add = shortcut and c1 == c2
       
           def forward(self, x):
               """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
               tensor.
               """
               return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
               
       class C3DCN(nn.Module):
           # CSP Bottleneck with 3 convolutions
           def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
               """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
               convolutions, and expansion.
               """
               super().__init__()
               c_ = int(c2 * e)  # hidden channels
               self.cv1 = Conv(c1, c_, 1, 1)
               self.cv2 = Conv(c1, c_, 1, 1)
               self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
               self.m = nn.Sequential(*(BottleneckDCN(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
       
           def forward(self, x):
               """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
               return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
       ```

    2. 修改models/yolov5s.yaml

       ```yaml
       # YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
       
       # Parameters
       nc: 80 # number of classes
       depth_multiple: 0.33 # model depth multiple
       width_multiple: 0.50 # layer channel multiple
       anchors:
         - [10, 13, 16, 30, 33, 23] # P3/8
         - [30, 61, 62, 45, 59, 119] # P4/16
         - [116, 90, 156, 198, 373, 326] # P5/32
       
       # YOLOv5 v6.0 backbone
       backbone:
         # [from, number, module, args]
         [
           [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2
           [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
           [-1, 3, C3DCN, [128]],
           [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
           [-1, 6, C3DCN, [256]],
           [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
           [-1, 9, C3DCN, [512]],
           [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
           [-1, 3, C3DCN, [1024]],
           [-1, 1, SPPF, [1024, 5]], # 9
         ]
       
       # YOLOv5 v6.0 head
       head: [
           [-1, 1, Conv, [512, 1, 1]],
           [-1, 1, nn.Upsample, [None, 2, "nearest"]],
           [[-1, 6], 1, Concat, [1]], # cat backbone P4
           [-1, 3, C3DCN, [512, False]], # 13
       
           [-1, 1, Conv, [256, 1, 1]],
           [-1, 1, nn.Upsample, [None, 2, "nearest"]],
           [[-1, 4], 1, Concat, [1]], # cat backbone P3
           [-1, 3, C3DCN, [256, False]], # 17 (P3/8-small)
       
           [-1, 1, Conv, [256, 3, 2]],
           [[-1, 14], 1, Concat, [1]], # cat head P4
           [-1, 3, C3DCN, [512, False]], # 20 (P4/16-medium)
       
           [-1, 1, Conv, [512, 3, 2]],
           [[-1, 10], 1, Concat, [1]], # cat head P5
           [-1, 3, C3DCN, [1024, False]], # 23 (P5/32-large)
       
           [[17, 20, 23], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)
         ]
       ```

    3. 同上

    4. 同上

- **添加注意力机制**

  - 添加BMA注意力机制

    1. models/common.py添加BMA模块

       ```python
       class BAMLayer(nn.Module):
           def __init__(self, gate_channel,reduction_ratio=16, dilation_val=4):
               super(BAMLayer, self).__init__()
               self.spatial_att = nn.Sequential(
                   nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1,bias=False),
                   nn.BatchNorm2d(gate_channel // reduction_ratio),
                   nn.ReLU(),
                   nn.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio,kernel_size=3,padding=dilation_val, dilation=dilation_val,bias=False),
                   nn.BatchNorm2d(gate_channel // reduction_ratio),
                   nn.ReLU(),
                   nn.Conv2d(gate_channel // reduction_ratio, gate_channel // reduction_ratio,kernel_size=3,padding=dilation_val, dilation=dilation_val,bias=False),
                   nn.BatchNorm2d(gate_channel // reduction_ratio),
                   nn.ReLU(),
                   nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1)
               )
               self.channel_att = nn.Sequential(
                   nn.Flatten(),
                   nn.Linear(gate_channel,gate_channel//reduction_ratio,bias=False),
                   nn.BatchNorm1d(gate_channel//reduction_ratio),
                   nn.ReLU(),
                   nn.Linear(gate_channel//reduction_ratio, gate_channel)
               )
           def forward(self,x):
               spatial_x = self.spatial_att(x).expand_as(x)
               avg_pool = F.avg_pool2d(x, x.size(2), x.size(2))
               channel_x = self.channel_att(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(x)
               return x + F.sigmoid(spatial_x + channel_x)*x
       ```

    2. 修改models/yolov5s.yaml

       ```yaml
       # YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
       
       # Parameters
       nc: 80 # number of classes
       depth_multiple: 0.33 # model depth multiple
       width_multiple: 0.50 # layer channel multiple
       anchors:
         - [10, 13, 16, 30, 33, 23] # P3/8
         - [30, 61, 62, 45, 59, 119] # P4/16
         - [116, 90, 156, 198, 373, 326] # P5/32
       
       # YOLOv5 v6.0 backbone
       backbone:
         # [from, number, module, args]
         [
           [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2
           [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
           [-1, 3, C3, [128]],
       
           [-1, 1, BAMLayer, [128,2,4]],
       
           [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
           [-1, 6, C3, [256]],
       
           [-1, 1, BAMLayer, [256,4,4]],
       
           [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
           [-1, 9, C3, [512]], # 6->8
       
           [-1, 1, BAMLayer, [512,8,4]],
       
           [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
           [-1, 3, C3, [1024]],
       
           [-1, 1, BAMLayer, [1024,16,4]],
       
           [-1, 1, SPPF, [1024, 5]], # 9
         ]
       
       # YOLOv5 v6.0 head
       head: [
           [-1, 1, Conv, [512, 1, 1]],
           [-1, 1, nn.Upsample, [None, 2, "nearest"]],
           [[-1, 9], 1, Concat, [1]], # cat backbone P4
           [-1, 3, C3, [512, False]], # 13
       
           [-1, 1, BAMLayer, [512,8,4]],
       
           [-1, 1, Conv, [256, 1, 1]],
           [-1, 1, nn.Upsample, [None, 2, "nearest"]],
           [[-1, 6], 1, Concat, [1]], # cat backbone P3
           [-1, 3, C3, [256, False]], # 17 (P3/8-small)
       
           [-1, 1, BAMLayer, [128,4,4]],
       
           [-1, 1, Conv, [256, 3, 2]],
           [[-1, 19], 1, Concat, [1]], # cat head P4
           [-1, 3, C3, [512, False]], # 20 (P4/16-medium)
       
           [-1, 1, BAMLayer, [128,2,4]],
       
           [-1, 1, Conv, [512, 3, 2]],
           [[-1, 14], 1, Concat, [1]], # cat head P5
           [-1, 3, C3, [1024, False]], # 23 (P5/32-large)
       
           [-1, 1, BAMLayer, [128,16,4]],
       
           [[23, 27, 31], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)
         ]
       
       ```

    3. 同上

    4. 同上

  - 添加CBMA注意力机制

    1. models/common.py添加CBMA模块

       ```python
       class CBAMLayer(nn.Module):
           def __init__(self, gate_channel,reduction_ratio=16, spatial_kernel=7):
               super(CBAMLayer, self).__init__()
               self.spatial_att = nn.Sequential(
                   nn.Conv2d(2, 1, kernel_size=spatial_kernel,padding=spatial_kernel//2),
               )
       
               self.channel_att = nn.Sequential(
                   nn.Flatten(),
                   nn.Linear(gate_channel, gate_channel // reduction_ratio),
                   nn.ReLU(),
                   nn.Linear(gate_channel // reduction_ratio, gate_channel)
               )
           def forward(self,x):
               spatial_x = F.sigmoid(self.spatial_att(torch.cat((torch.max(x,dim=1,keepdim=True)[0],torch.mean(x,dim=1,keepdim=True)),dim=1))).expand_as(x)
               avg_pool = F.avg_pool2d(x, x.size(2), x.size(2))
               max_pool = F.max_pool2d(x, x.size(2), x.size(2))
               avg_x = self.channel_att(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(x)
               max_x = self.channel_att(max_pool).unsqueeze(2).unsqueeze(3).expand_as(x)
               channel_x = F.sigmoid(avg_x+max_x)
       
               return (x*channel_x)*(x*channel_x)*spatial_x
       ```

    2. 修改models/yolov5s.yaml

       ```yaml
       # YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
       
       # Parameters
       nc: 80 # number of classes
       depth_multiple: 0.33 # model depth multiple
       width_multiple: 0.50 # layer channel multiple
       anchors:
         - [10, 13, 16, 30, 33, 23] # P3/8
         - [30, 61, 62, 45, 59, 119] # P4/16
         - [116, 90, 156, 198, 373, 326] # P5/32
       
       # YOLOv5 v6.0 backbone
       backbone:
         # [from, number, module, args]
         [
           [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2
           [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
           [-1, 3, C3, [128]],
       
           [-1, 1, CBAMLayer, [128,2,7]],
       
           [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
           [-1, 6, C3, [256]],
       
           [-1, 1, CBAMLayer, [256,4,7]],
       
           [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
           [-1, 9, C3, [512]], # 6->8
       
           [-1, 1, CBAMLayer, [512,8,7]],
       
           [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
           [-1, 3, C3, [1024]],
       
           [-1, 1, CBAMLayer, [1024,16,7]],
       
           [-1, 1, SPPF, [1024, 5]], # 9
         ]
       
       # YOLOv5 v6.0 head
       head: [
           [-1, 1, Conv, [512, 1, 1]],
           [-1, 1, nn.Upsample, [None, 2, "nearest"]],
           [[-1, 9], 1, Concat, [1]], # cat backbone P4
           [-1, 3, C3, [512, False]], # 13
       
           [-1, 1, CBAMLayer, [512,8,7]],
       
           [-1, 1, Conv, [256, 1, 1]],
           [-1, 1, nn.Upsample, [None, 2, "nearest"]],
           [[-1, 6], 1, Concat, [1]], # cat backbone P3
           [-1, 3, C3, [256, False]], # 17 (P3/8-small)
       
           [-1, 1, CBAMLayer, [128,4,7]],
       
           [-1, 1, Conv, [256, 3, 2]],
           [[-1, 19], 1, Concat, [1]], # cat head P4
           [-1, 3, C3, [512, False]], # 20 (P4/16-medium)
       
           [-1, 1, CBAMLayer, [128,2,7]],
       
           [-1, 1, Conv, [512, 3, 2]],
           [[-1, 14], 1, Concat, [1]], # cat head P5
           [-1, 3, C3, [1024, False]], # 23 (P5/32-large)
       
           [-1, 1, CBAMLayer, [128,16,7]],
       
           [[23, 27, 31], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)
         ]
       
       ```

    3. 同上

    4. 同上

- **添加neck**

  - 添加ASFF
    1. 

- **替换iou**

- **替换nms**

- **替换样本分配策略**

- **添加辅助训练分支**

- **替换解码输出头**

- **添加输出头**



​		