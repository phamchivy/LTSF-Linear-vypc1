import torch
import torch.nn as nn

class SVDLowRank(nn.Module):

    def __init__(self, rank=2):
        super().__init__()
        self.rank = rank

    def forward(self, x):
        # x: [B,L,C]

        B,L,C = x.shape
        outputs=[]

        for i in range(B):

            U,S,Vh = torch.linalg.svd(
                x[i],
                full_matrices=False
            )

            k=min(self.rank,S.shape[0])

            x_lowrank = (
                U[:,:k]
                @ torch.diag(S[:k])
                @ Vh[:k,:]
            )

            outputs.append(x_lowrank)

        return torch.stack(outputs,0)