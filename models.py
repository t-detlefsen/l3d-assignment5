import torch
import torch.nn as nn
import torch.nn.functional as F

class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Block 2
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Block 3
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        
        self.cls = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        x = points.transpose(2,1)
        x = self.backbone(x)
        x = F.max_pool1d(x, points.shape[1]).squeeze(-1)
        x = self.cls(x)

        return x

# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        
        self.backbone1 = nn.Sequential(
            # Block 1
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.backbone2 = nn.Sequential(
            # Block 2
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Block 3
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.seg = nn.Sequential(
            # Block 1
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # Block 2
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # Block 3
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Output
            nn.Conv1d(128, num_seg_classes, 1),
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        x = points.transpose(2,1)
        local_features = self.backbone1(x)
        x = self.backbone2(local_features)
        global_features = F.max_pool1d(x, points.shape[1]).squeeze(-1)
        global_features = global_features.unsqueeze(2).repeat(1, 1, points.shape[1])

        x = torch.cat((local_features, global_features), 1)
        x = self.seg(x)
        x = x.transpose(2,1)

        return x