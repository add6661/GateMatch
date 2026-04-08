import torch
import torch.nn as nn
import torch.nn.functional as F


class GridPosition(nn.Module):
    def __init__(self, grid_num, use_gpu = True):
        nn.Module.__init__(self)
        self.grid_num = grid_num
        self.use_gpu = use_gpu

    def forward(self, batch_size):
        grid_center_x = torch.linspace(-1.+2./self.grid_num/2,1.-2./self.grid_num/2,steps=self.grid_num).cuda() if self.use_gpu else torch.linspace(-1.+1./self.grid_num/2,1.-1./self.grid_num/2,steps=self.grid_num)
        grid_center_y = torch.linspace(1.-2./self.grid_num/2,-1.+2./self.grid_num/2,steps=self.grid_num).cuda() if self.use_gpu else torch.linspace(1.-1./self.grid_num/2,-1.+1./self.grid_num/2,steps=self.grid_num)
        # BCHW, (b,:,h,w)->(x,y)
        grid_center_position_mat = torch.reshape(
            torch.cartesian_prod(grid_center_x, grid_center_y),
            (1, self.grid_num, self.grid_num, 2)
        ).permute(0,3,2,1)
        # BCN, (b,:,n)->(x,y), left to right then up to down
        grid_center_position_seq = grid_center_position_mat.reshape(1, 2, self.grid_num*self.grid_num)
        return grid_center_position_seq.repeat(batch_size, 1, 1)


class AttentionPropagation(nn.Module):
    def __init__(self, channels, head):
        nn.Module.__init__(self)
        self.head = head
        self.head_dim = channels // head
        self.query_filter, self.key_filter, self.value_filter = nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1)
        self.mh_filter = nn.Conv1d(channels, channels, kernel_size=1)
        self.cat_filter = nn.Sequential(
            nn.Conv1d(2*channels, 2*channels, kernel_size=1),
            nn.BatchNorm1d(2*channels), nn.ReLU(),
            nn.Conv1d(2*channels, channels, kernel_size=1),
        )

    def forward(self, motion1, motion2):
        # motion1(q) attend to motion(k,v)
        batch_size = motion1.shape[0]
        query, key, value = self.query_filter(motion1).view(batch_size, self.head, self.head_dim, -1),\
                            self.key_filter(motion2).view(batch_size, self.head, self.head_dim, -1),\
                            self.value_filter(motion2).view(batch_size, self.head, self.head_dim, -1)
        score = torch.softmax(torch.einsum('bhdn,bhdm->bhnm', query, key) / self.head_dim ** 0.5, dim = -1)
        add_value = torch.einsum('bhnm,bhdm->bhdn', score, value).reshape(batch_size, self.head_dim * self.head, -1)
        add_value = self.mh_filter(add_value)
        motion1_new = motion1 + self.cat_filter(torch.cat([motion1, add_value], dim=1))
        return motion1_new



class SEAttention(nn.Module):

    def __init__(self, channels, reduction=4):
        nn.Module.__init__(self)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CoordinateAttention(nn.Module):

    def __init__(self, channels, reduction=4):
        nn.Module.__init__(self)
        self.channels = channels
        mid_channels = max(8, channels // reduction)
        
        self.conv_encode = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_h = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv_encode(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        
        out = x * a_h * a_w
        return out


class GlobalGeometricConsistencyModule(nn.Module):

    def __init__(self, channels, reduction=4):
        nn.Module.__init__(self)
        self.channels = channels
        
        self.coord_attention = CoordinateAttention(channels, reduction)
        self.se_attention = SEAttention(channels, reduction)
        
        self.complexity_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 1),
            nn.Sigmoid()
        )
        
        self.geometry_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):

        feat_var = x.var(dim=[2, 3], keepdim=True)
        complexity_input = x * feat_var  
        gate = self.complexity_gate(complexity_input)  # (B, 1)
        gate = gate.view(-1, 1, 1, 1)
        
        coord_feat = self.coord_attention(x)
        se_feat = self.se_attention(x)
        
        attention_feat = gate * se_feat + (1 - gate) * coord_feat

        geo_feat = self.geometry_enhance(x)

        out = attention_feat + geo_feat + x
        
        return out


class MultiScaleDilatedPerceptionModule(nn.Module):

    def __init__(self, channels):
        nn.Module.__init__(self)
        branch_channels = channels // 3

        self.conv_small = nn.Sequential(
            nn.Conv2d(channels, branch_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_medium = nn.Sequential(
            nn.Conv2d(channels, branch_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_large = nn.Sequential(
            nn.Conv2d(channels, branch_channels, 3, padding=4, dilation=4),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.scale_gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 3, 1),  
        )

        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.edge_refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

        concat_channels = branch_channels * 3
        self.fuse = nn.Sequential(
            nn.Conv2d(concat_channels, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        b, c, h, w = x.size()

        small_feat = self.conv_small(x)
        medium_feat = self.conv_medium(x)
        large_feat = self.conv_large(x)

        scale_weights = self.scale_gate(x)  # (B, 3, H, W)
        scale_weights = F.softmax(scale_weights, dim=1)  

        w_s, w_m, w_l = scale_weights[:, 0:1], scale_weights[:, 1:2], scale_weights[:, 2:3]
        weighted_small = small_feat * w_s
        weighted_medium = medium_feat * w_m
        weighted_large = large_feat * w_l
        
        multi_scale = torch.cat([weighted_small, weighted_medium, weighted_large], dim=1)

        x_gray = x.mean(dim=1, keepdim=True)  
        grad_x = F.conv2d(x_gray, self.sobel_x.expand(1, 1, 3, 3), padding=1)
        grad_y = F.conv2d(x_gray, self.sobel_y.expand(1, 1, 3, 3), padding=1)
        edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        edge_weight = self.edge_refine(edge_magnitude)

        out = self.fuse(multi_scale)

        out = edge_weight * x + (1 - edge_weight) * out + x
        
        return out


class OriginalMotionReferenceModule(nn.Module):

    def __init__(self, channels, grid_num=16):
        nn.Module.__init__(self)
        self.channels = channels

        self.orig_motion_encoder = nn.Sequential(
            nn.Conv1d(2, channels // 2, kernel_size=1), nn.BatchNorm1d(channels // 2), nn.ReLU(),
            nn.Conv1d(channels // 2, channels, kernel_size=1)
        )
        
        self.local_consistency = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=1),
            nn.BatchNorm1d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 2, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )

        self.consistency_confidence = nn.Sequential(
            nn.Conv1d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.contrast_module = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        self.correction_gate = nn.Sequential(
            nn.Conv1d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.multi_level_fuse = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, current_feat, original_motion, init_feat):

        b, c, n = current_feat.shape

        orig_motion_feat = self.orig_motion_encoder(original_motion)

        local_consistent_feat = self.local_consistency(current_feat)

        feat_diff = (current_feat - local_consistent_feat).abs()
        consistency_conf = self.consistency_confidence(feat_diff)  # (B, 1, N)

        contrast_input = torch.cat([current_feat, orig_motion_feat], dim=1)
        contrast_residual = self.contrast_module(contrast_input)

        adjusted_contrast = contrast_residual * consistency_conf

        cumulative_correction = current_feat - init_feat
        gate = self.correction_gate(cumulative_correction)

        multi_level_feat = self.multi_level_fuse(
            torch.cat([orig_motion_feat, init_feat, current_feat], dim=1)
        )

        enhanced_feat = gate * current_feat + (1 - gate) * (current_feat + adjusted_contrast)

        enhanced_feat = enhanced_feat + 0.1 * local_consistent_feat + 0.1 * multi_level_feat
        
        return enhanced_feat, contrast_residual


class ResBlock(nn.Module):

    def __init__(self, channels):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.net(x) + x
        return x


class EnhancedFilterBlock(nn.Module):

    def __init__(self, channels, use_ggcm=True, use_msdpm=True):
        nn.Module.__init__(self)
        self.use_ggcm = use_ggcm
        self.use_msdpm = use_msdpm
        
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        
        if use_ggcm:
            self.ggcm = GlobalGeometricConsistencyModule(channels)
        if use_msdpm:
            self.msdpm = MultiScaleDilatedPerceptionModule(channels)

        num_branches = int(use_ggcm) + int(use_msdpm)
        if num_branches == 0:
            self.fuse = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        elif num_branches == 1:
            self.fuse = nn.Sequential(
                nn.Conv2d(channels, channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.fuse = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        local_feat = self.local_conv(x)
        
        features = []
        if self.use_ggcm:
            features.append(self.ggcm(local_feat))
        if self.use_msdpm:
            features.append(self.msdpm(local_feat))
        
        if len(features) == 0:
            out = self.fuse(local_feat)
        elif len(features) == 1:
            out = self.fuse(features[0])
        else:
            out = self.fuse(torch.cat(features, dim=1))
        
        return out + x


class Filter(nn.Module):
    def __init__(self, channels, use_ggcm=True, use_msdpm=True):
        nn.Module.__init__(self)
        self.resnet = nn.Sequential(*[
            EnhancedFilterBlock(channels, use_ggcm, use_msdpm) for _ in range(3)
        ])
        self.scale = nn.Sequential(
            nn.Conv2d(channels, channels, 1, padding=0),
        )

    def forward(self, x):
        x = self.scale(self.resnet(x))
        return x


class FilterNet(nn.Module):
    def __init__(self, grid_num, channels, use_ggcm=True, use_msdpm=True):
        nn.Module.__init__(self)
        self.grid_num = grid_num
        self.filter = Filter(channels, use_ggcm, use_msdpm)

    def forward(self, x):
        # BCN -> BCHW
        x = x.reshape(x.shape[0], x.shape[1], self.grid_num, self.grid_num)
        x = self.filter(x)
        # BCHW -> BCN
        x = x.reshape(x.shape[0], x.shape[1], self.grid_num*self.grid_num)
        return x


class PositionEncoder(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.position_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, channels, kernel_size=1)
        )
        
    def forward(self, x):
        return self.position_encoder(x)


class InitProject(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.init_project = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, channels, kernel_size=1)
        )
        
    def forward(self, x):
        return self.init_project(x)


class InlinerPredictor(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.inlier_pre = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=1), nn.InstanceNorm1d(64, eps=1e-3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 16, kernel_size=1), nn.InstanceNorm1d(16, eps=1e-3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 4, kernel_size=1), nn.InstanceNorm1d(4, eps=1e-3), nn.BatchNorm1d(4), nn.ReLU(),
            nn.Conv1d(4, 1, kernel_size=1)
        )

    def forward(self, d):
        # BCN -> B1N
        return self.inlier_pre(d)


class EnhancedInlierPredictor(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.layer_residual_branch = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=1), nn.InstanceNorm1d(64, eps=1e-3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=1), nn.InstanceNorm1d(32, eps=1e-3), nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.contrast_residual_branch = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=1), nn.InstanceNorm1d(64, eps=1e-3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=1), nn.InstanceNorm1d(32, eps=1e-3), nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.final_predict = nn.Sequential(
            nn.Conv1d(64, 16, kernel_size=1), nn.InstanceNorm1d(16, eps=1e-3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 4, kernel_size=1), nn.InstanceNorm1d(4, eps=1e-3), nn.BatchNorm1d(4), nn.ReLU(),
            nn.Conv1d(4, 1, kernel_size=1)
        )

    def forward(self, layer_residual, contrast_residual):
        layer_feat = self.layer_residual_branch(layer_residual)
        contrast_feat = self.contrast_residual_branch(contrast_residual)
        combined = torch.cat([layer_feat, contrast_feat], dim=1)
        return self.final_predict(combined)


class LayerBlock(nn.Module):
    def __init__(self, channels, head, grid_num, use_ggcm=True, use_msdpm=True, use_omrm=True):
        nn.Module.__init__(self)
        self.use_omrm = use_omrm
        self.grid_num = grid_num
        
        self.align = AttentionPropagation(channels, head)
        self.filter = FilterNet(grid_num, channels, use_ggcm, use_msdpm)
        self.dealign = AttentionPropagation(channels, head)
        if use_omrm:
            self.omrm = OriginalMotionReferenceModule(channels, grid_num)  # 传递grid_num
            self.inlier_pre = EnhancedInlierPredictor(channels)
        else:
            self.inlier_pre = InlinerPredictor(channels)

    def forward(self, xs, d, grid_pos_embed, original_motion=None, init_feat=None):
        # xs: B1N4
        grid_d = self.align(grid_pos_embed, d)
        grid_d = self.filter(grid_d)
        d_new = self.dealign(d, grid_d)
        
        if self.use_omrm:
            d_enhanced, contrast_residual = self.omrm(d_new, original_motion, init_feat)
            layer_residual = d_enhanced - d
            logits = torch.squeeze(self.inlier_pre(layer_residual, contrast_residual), 1)
            d_out = d_enhanced
        else:
            logits = torch.squeeze(self.inlier_pre(d_new - d), 1)
            d_out = d_new
        
        e_hat = weighted_8points(xs, logits)
        
        return d_out, logits, e_hat


class GateMatch(nn.Module):
    def __init__(self, config, use_gpu=True):
        nn.Module.__init__(self)
        self.layer_num = config.layer_num
        self.use_ggcm = getattr(config, 'use_ggcm', True)
        self.use_msdpm = getattr(config, 'use_msdpm', True)
        self.use_omrm = getattr(config, 'use_omrm', True)
        print(f"[GateMatch Enhanced] GGCM: {self.use_ggcm}, MSDPM: {self.use_msdpm}, OMRM: {self.use_omrm}")

        self.grid_center = GridPosition(config.grid_num, use_gpu=use_gpu)
        self.pos_embed = PositionEncoder(config.net_channels)
        self.grid_pos_embed = PositionEncoder(config.net_channels)
        self.init_project = InitProject(config.net_channels)
        self.layer_blocks = nn.Sequential(
            *[LayerBlock(
                config.net_channels, config.head, config.grid_num,
                self.use_ggcm, self.use_msdpm, self.use_omrm
            ) for _ in range(self.layer_num)]
        )

    def forward(self, data):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        # B1NC -> BCN
        input = data['xs'].transpose(1,3).squeeze(3)
        x1, x2 = input[:,:2,:], input[:,2:,:]
        motion = x2 - x1 

        pos = x1 # B2N
        grid_pos = self.grid_center(batch_size) # B2N

        pos_embed = self.pos_embed(pos) # BCN
        grid_pos_embed = self.grid_pos_embed(grid_pos)

        d = self.init_project(motion) + pos_embed # BCN

        init_feat = d.clone() if self.use_omrm else None

        res_logits, res_e_hat = [], []
        for i in range(self.layer_num):
            if self.use_omrm:
                d, logits, e_hat = self.layer_blocks[i](
                    data['xs'], d, grid_pos_embed, motion, init_feat
                )
            else:
                d, logits, e_hat = self.layer_blocks[i](
                    data['xs'], d, grid_pos_embed
                )
            res_logits.append(logits), res_e_hat.append(e_hat)
        return res_logits, res_e_hat 


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.linalg.eigh(X[batch_idx,:,:].squeeze(), UPLO='L')
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)
    
    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)
    

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat
