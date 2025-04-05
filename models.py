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
            F.relu(),
            # Block 2
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            F.relu(),
            # Block 3
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            F.relu(),
            # Max Pooling
            nn.MaxPool1d(1024)
        )
        
        self.cls = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            F.relu(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            F.relu(),
            nn.Linear(256, 9),
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        import ipdb
        ipdb.set_trace()
        x = self.backbone(points)
        x = self.cls(x)

        return F.one_hot(x)



# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        pass

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        pass



