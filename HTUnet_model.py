import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import trunc_normal_
import math


class Mlp(nn.Module):
    """
    This class defines the Feed Forward Network (Multilayer perceptron)
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Self_Attention(nn.Module):
    def __init__(self, dim, ratio_h=2, ratio_w=2, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """This class defines the self-attention utilized in the Efficient Transformer block used in the global branch of the META module

        Args:
            dim (_type_): the number of the channel dimension of the input feature map.
            ratio_h (int, optional): the reduction ratio of the height used in efficient transformer block. Defaults to 2.
            ratio_w (int, optional): the reduction ratio of the width used in efficient transformer block.. Defaults to 2.
            num_heads (int, optional): the number of the heads used in multi-head self-attention. Defaults to 8.
            qkv_bias (bool, optional): whether use bias in the linear layer projecting Q, K, and V. Defaults to False.
            qk_scale (_type_, optional): Defaults to None.
            attn_drop (_type_, optional): Defaults to 0..
            proj_drop (_type_, optional): Defaults to 0..
        """
        super().__init__()
        self.s = int(ratio_h * ratio_w)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.ke = nn.Conv2d(dim, dim, kernel_size=(ratio_h, ratio_w), stride=(ratio_h, ratio_w), bias=qkv_bias)
        self.ve = nn.Conv2d(dim, dim, kernel_size=(ratio_h, ratio_w), stride=(ratio_h, ratio_w), bias=qkv_bias)
        self.norm_k = nn.LayerNorm(head_dim)
        self.norm_v = nn.LayerNorm(head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # print("x,shape",x.shape)
        H = W = int(math.sqrt(N))
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        # print("k.shape",(qkv[1].transpose(1, 2).reshape(B, C, H, W).flatten(2).transpose(1, 2)).shape)
        # print(N,self.s,N // self.s)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.ke(k.transpose(1, 2).reshape(B, C, H, W)).flatten(2).transpose(1, 2).reshape(B, N // self.s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.ve(v.transpose(1, 2).reshape(B, C, H, W)).flatten(2).transpose(1, 2).reshape(B, N // self.s, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.norm_k(k)
        v = self.norm_v(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ETransformer_block(nn.Module):

    def __init__(self, dim, ratio_h=2, ratio_w=2, num_heads=8, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, out_features=None, mlp_ratio=4.,):
        """This class defines the Efficient Transformer block used in the global branch of the META module

        Args:
            dim (_type_): the number of the channel dimension of the input feature map.
            ratio_h (int, optional): the reduction ratio of the height used in efficient transformer block. Defaults to 2.
            ratio_w (int, optional): the reduction ratio of the width used in efficient transformer block.. Defaults to 2.
            num_heads (int, optional): the number of the heads used in multi-head self-attention. Defaults to 8.
            qkv_bias (bool, optional): whether use bias in the linear layer projecting Q, K, and V. Defaults to False.
            qk_scale (_type_, optional): Defaults to None.
            attn_drop (_type_, optional): Defaults to 0..
            proj_drop (_type_, optional): Defaults to 0..
            act_layer (_type_, optional): the action function used in FFN. Defaults to nn.GELU.
            norm_layer (_type_, optional): Defaults to nn.LayerNorm.
            out_features (_type_, optional): Defaults to None.
            mlp_ratio (_type_, optional): Defaults to 4..
        """
        super().__init__()
        self.out_features = out_features
        self.norm1 = norm_layer(dim)
        self.attn = Self_Attention(
            dim, ratio_h, ratio_w, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_features, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        if self.out_features:
            return self.mlp(self.norm2(x))
        else:
            return x + self.mlp(self.norm2(x))

class Self_Attention_local(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """This class defines the self-attention utilized in the Efficient Transformer block used in the local branch of the META module

        Args:
            dim (_type_): the number of the channel dimension of the input feature map.
            num_heads (int, optional): the number of the heads used in multi-head self-attention. Defaults to 8.
            qkv_bias (bool, optional): whether use bias in the linear layer projecting Q, K, and V. Defaults to False.
            qk_scale (_type_, optional): Defaults to None.
            attn_drop (_type_, optional): Defaults to 0..
            proj_drop (_type_, optional): Defaults to 0..
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, R, N, C = x.shape
        qkv = self.qkv(x).reshape(B, R, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(-1, -2).reshape(B, R, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ETransformer_block_local(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, num_heads=8, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, out_features=None, mlp_ratio=4.,):
        """This class defines the Efficient Transformer block used in the local branch of the META module

        Args:
            dim (_type_): the number of the channel dimension of the input feature map.
            num_heads (int, optional): the number of the heads used in multi-head self-attention. Defaults to 8.
            qkv_bias (bool, optional): whether use bias in the linear layer projecting Q, K, and V. Defaults to False.
            qk_scale (_type_, optional): Defaults to None.
            attn_drop (_type_, optional): Defaults to 0..
            drop (_type_, optional): Defaults to 0..
            act_layer (_type_, optional): Defaults to nn.GELU.
            norm_layer (_type_, optional): Defaults to nn.LayerNorm.
            out_features (_type_, optional): Defaults to None.
            mlp_ratio (_type_, optional): Defaults to 4..
        """
        super().__init__()
        self.out_features = out_features
        self.norm1 = norm_layer(dim)
        self.attn = Self_Attention_local(
            dim, qkv_bias=qkv_bias, qk_scale=qk_scale, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_features, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        if self.out_features:
            return self.mlp(self.norm2(x))
        else:
            return x + self.mlp(self.norm2(x))

class META(nn.Module):
    def __init__(self, dim, ph=5, pw=5, ratio_h=2, ratio_w=2, num_heads=8, drop=0., attn_drop=0.):
        """this class defines the Multiscale Efficient Transformer Attention module

        Args:
            dim (_type_): the number of the channel dimension of the input feature map.
            ph (int, optional): the patch size of height in the local branch. Defaults to 4.
            pw (int, optional): the patch size of width in the local branch. Defaults to 4.
            ratio_h (int, optional): the reduction ratio of the height used in efficient transformer block. Defaults to 2.
            ratio_w (int, optional): the reduction ratio of the width used in efficient transformer block.. Defaults to 2.
            num_heads (int, optional): Defaults to 8.
            drop (_type_, optional): Defaults to 0..
            attn_drop (_type_, optional): Defaults to 0..
        """
        super().__init__()
        self.ph = ph
        self.pw = pw
        self.loc_attn = ETransformer_block_local(dim=dim, num_heads=num_heads, drop=drop, attn_drop=attn_drop)
        self.glo_attn = ETransformer_block(dim=dim, ratio_h=ratio_h, ratio_w=ratio_w, num_heads=num_heads, drop=drop, attn_drop=attn_drop)

    def forward(self, x, feature=False):
        b, c, h, w = x.shape
        loc_x = rearrange(x, 'b d (nh ph) (nw pw) -> b (nh nw) (ph pw) d', ph=self.ph, pw=self.pw)
        glo_x = x.flatten(2).transpose(1, 2)
        loc_y = self.loc_attn(loc_x)
        loc_y = rearrange(loc_y, 'b (nh nw) (ph pw) d -> b d (nh ph) (nw pw)', nh=h // self.ph, nw=w // self.pw,
                          ph=self.ph, pw=self.pw)
        glo_y = self.glo_attn(glo_x)
        glo_y = glo_y.transpose(1, 2).reshape(b, c, h, w)
        y = loc_y + glo_y
        y = torch.sigmoid(y)
        if feature:
            return loc_y, glo_y, x * y
        else:
            return x * y

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, 
                      stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # print(x.size())
            # print(self.weight[:, None, None].size())
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    

class group_aggregation_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1,2,5,7]):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        # print("group_size",group_size,dim_xl)
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[0]-1))//2, 
                      dilation=d_list[0], groups=group_size + 1)
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[1]-1))//2, 
                      dilation=d_list[1], groups=group_size + 1)
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[2]-1))//2, 
                      dilation=d_list[2], groups=group_size + 1)
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[3]-1))//2, 
                      dilation=d_list[3], groups=group_size + 1)
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 4, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + 4, dim_xl, 1)
        )
    def forward(self, xh, xl, mask):
        # print('--------------------xh',xh.size())
        xh = self.pre_project(xh)
        # print('--------------------xh',xh.size())
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode ='bilinear', align_corners=True)
        # print('--------------------xh',xh.size())
        xh = torch.chunk(xh, 4, dim=1)
        # print('--------------------xh',xh.size())
        xl = torch.chunk(xl, 4, dim=1)
        # print(len(xh[0]),len(xh[1]),len(xh[2]),len(xh[3]))
        # print(len(xl[0]),len(xl[1]),len(xl[2]),len(xl[3]))
        # print(len(mask[0]),len(mask[1]),len(mask[2]),len(mask[3]),len(mask[4]),len(mask[5]),len(mask[6]),len(mask[7]))
        # print(xl)
        # print(mask)
        # print(len(xh),len(xl),len(mask))
        # print("size",xh[0].size(),xl[0].size(),mask.size())    #torch.Size([8, 8, 25, 25]) torch.Size([8, 8, 25, 25]) torch.Size([8, 1, 24, 24])
        x0 = self.g0(torch.cat((xh[0], xl[0], mask), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1], mask), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2], mask), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3], mask), dim=1))
        x = torch.cat((x0,x1,x2,x3), dim=1)
        x = self.tail_conv(x)
        return x


class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()
        
        c_dim_in = dim_in//4
        k_size=3
        pad=(k_size-1) // 2
        
        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
                nn.Conv2d(c_dim_in, c_dim_in, 1),
                nn.GELU(),
                nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )
        
        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        
        self.ldw = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
                nn.GELU(),
                nn.Conv2d(dim_in, dim_out, 1),
        )
        
    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C, H, W = x1.size()
        #----------xy----------#
        params_xy = self.params_xy
        x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4],mode='bilinear', align_corners=True))
        #----------zx----------#
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(F.interpolate(params_zx, size=x2.shape[2:4],mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        #----------zy----------#
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(F.interpolate(params_zy, size=x3.shape[2:4],mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        #----------dw----------#
        x4 = self.dw(x4)
        #----------concat----------#
        x = torch.cat([x1,x2,x3,x4],dim=1)
        #----------ldw----------#
        x = self.norm2(x)
        x = self.ldw(x)
        return x



    
    

class HTUnet(nn.Module):
    
    def __init__(self, num_classes=1, input_channels=35, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True,p1=5, p2=5, p3=3):
        super().__init__()

        self.bridge = bridge
        self.gt_ds = gt_ds
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[0], c_list[1]),
        ) 
        self.encoder3 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[1], c_list[2]),
        )
        self.encoder4 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[2], c_list[3]),
        )
        self.encoder5 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[4]),
        )
        self.encoder6 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[5]),
        )

        if bridge: 
            self.GAB1 = group_aggregation_bridge(c_list[1], c_list[0])
            self.GAB2 = group_aggregation_bridge(c_list[2], c_list[1])
            self.GAB3 = group_aggregation_bridge(c_list[3], c_list[2])
            self.GAB4 = group_aggregation_bridge(c_list[4], c_list[3])
            self.GAB5 = group_aggregation_bridge(c_list[5], c_list[4])
            print('group_aggregation_bridge was used')
        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))
            print('gt deep supervision was used')

        num_heads = 4
        self.Trans = META(dim=64, ph=p3, pw=p3, ratio_h=4, ratio_w=4, num_heads=num_heads, drop=0., attn_drop=0.)


        self.decoder1 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[5], c_list[4]),
        ) 
        self.decoder2 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[3]),
        ) 
        self.decoder3 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[2]),
        )  
        # self.decoder4 = nn.Sequential(
        #     nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        # )  
        self.decoder4 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[2], c_list[1]),
        ) 
        # self.decoder5 = nn.Sequential(
        #     nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        # )  
        self.decoder5 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[1], c_list[0]),
        )         
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out # b, c1, H/4, W/4 

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out # b, c2, H/8, W/8
        
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = out # b, c3, H/16, W/16
        
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        t5 = out # b, c4, H/32, W/32
        
        out = F.gelu(self.encoder6(out)) # b, c5, H/32, W/32
        t6 = out

        out65 = self.Trans(out)

        out5 = F.gelu(self.dbn1(self.decoder1(out65))) # b, c4, H/32, W/32
        # print("out5",out5.size())
        if self.gt_ds: 
            gt_pre5 = self.gt_conv1(out5)
            # print("gt_pre5",t6.size(),t5.size(),gt_pre5.size())
            # print("before   gt_pre5",gt_pre5.size())
            t5 = self.GAB5(t6, t5, gt_pre5)
            gt_pre5 = F.interpolate(gt_pre5, scale_factor=(400/12), mode ='bilinear', align_corners=True)
            # print("2======gt_pre5",gt_pre5.size())
        else: t5 = self.GAB5(t6, t5)
        out5 = torch.add(out5, t5) # b, c4, H/32, W/32
        # print("-after  out5",out5.size())
        # print("self.dbn2(self.decoder2(out5))",self.dbn2(self.decoder2(out5)).size())
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),size=[t4.size(2), t4.size(3)],mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16
        # print("------------  out4",out4.size())
        if self.gt_ds: 
            # print("out4",out4.size())
            gt_pre4 = self.gt_conv2(out4)
            # print("gt_pre4",t5.size(),t4.size(),gt_pre4.size())
            t4 = self.GAB4(t5, t4, gt_pre4)
            gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode ='bilinear', align_corners=True)
        else:t4 = self.GAB4(t5, t4)
        out4 = torch.add(out4, t4) # b, c3, H/16, W/16
        
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
        if self.gt_ds: 
            gt_pre3 = self.gt_conv3(out3)
            t3 = self.GAB3(t4, t3, gt_pre3)
            gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode ='bilinear', align_corners=True)
        else: t3 = self.GAB3(t4, t3)
        out3 = torch.add(out3, t3) # b, c2, H/8, W/8
        
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4
        if self.gt_ds: 
            gt_pre2 = self.gt_conv4(out2)
            t2 = self.GAB2(t3, t2, gt_pre2)
            gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode ='bilinear', align_corners=True)
        else: t2 = self.GAB2(t3, t2)
        out2 = torch.add(out2, t2) # b, c1, H/4, W/4 
        
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2
        if self.gt_ds: 
            gt_pre1 = self.gt_conv5(out1)
            t1 = self.GAB1(t2, t1, gt_pre1)
            gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode ='bilinear', align_corners=True)
        else: t1 = self.GAB1(t2, t1)
        out1 = torch.add(out1, t1) # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W
        # out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W
        
        if self.gt_ds:
            #relu = nn.ReLU()
            # print("(gt_pre5.size(),gt_pre4.size(),gt_pre3.size(),gt_pre2.size(),gt_pre1.size()",gt_pre5.size(),gt_pre4.size(),gt_pre3.size(),gt_pre2.size(),gt_pre1.size())
            return (torch.sigmoid(gt_pre5), torch.sigmoid(gt_pre4), torch.sigmoid(gt_pre3), torch.sigmoid(gt_pre2), torch.sigmoid(gt_pre1)), torch.sigmoid(out0)
            #return (relu(gt_pre5), relu(gt_pre4), relu(gt_pre3), relu(gt_pre2), relu(gt_pre1)), relu(out0)            
        else:
            return torch.sigmoid(out0)
