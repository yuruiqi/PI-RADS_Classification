import torch
import torch.nn as nn


class UnsureDataLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        device = config['DEVICE']
        # avoid inf
        self.bias = torch.tensor([1e-7], dtype=torch.float32, device=device, requires_grad=False)

        require_grad = True
        self.t23 = nn.Parameter(torch.tensor([-5], dtype=torch.float32), requires_grad=require_grad)
        self.t34 = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=require_grad)
        self.t45 = nn.Parameter(torch.tensor([5], dtype=torch.float32), requires_grad=require_grad)

        self.e_bool = config['E BOOL']
        if self.e_bool:
            # <1 to let t higher, or >1 to let t lower
            self.e23 = nn.Parameter(torch.tensor([-1], dtype=torch.float32), requires_grad=require_grad)
            self.e34 = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=require_grad)
            self.e45 = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=require_grad)

    def forward(self, pred, label):
        """

        :param pred: (batch, 1). Logits before sigmoid.
        :param label:  (batch, 1)
        :return:
        """
        if self.e_bool:
            corrs = [[self.t23, torch.sigmoid(self.e23)],
                     [self.t34, torch.sigmoid(self.e34)],
                     [self.t45, torch.sigmoid(self.e45)]]
        else:
            corrs = [[self.t23, ],
                     [self.t34, ],
                     [self.t45, ]]

        def compute_lossx(x):
            index = label.eq(x)
            if index.nonzero().shape[0] == 0:
                return torch.tensor(0, dtype=torch.float32, device=pred.device)
            else:
                pred_x = pred[index]  # (n, 1)
                if self.e_bool:
                    p_x = [torch.tensor([0], dtype=pred.dtype, device=pred.device), ] + \
                        [torch.sigmoid(-pred_x + t[0] + torch.log(t[1] + self.bias)) for t in corrs] + \
                        [torch.tensor([1], dtype=pred.dtype, device=pred.device), ]  # [0, p23 ,p34, p45, 1]
                else:
                    p_x = [torch.tensor([0], dtype=pred.dtype, device=pred.device), ] + \
                          [torch.sigmoid(-pred_x + t[0]) for t in corrs] + \
                          [torch.tensor([1], dtype=pred.dtype, device=pred.device), ]  # [0, p23 ,p34, p45, 1]

                loss_x = -torch.log(p_x[x-1]-p_x[x-2] + self.bias)
                if torch.isinf(loss_x).sum():
                    print(self.e23, self.e34, self.e45)
                return torch.sum(loss_x)/pred.shape[0]

        loss = compute_lossx(5) + compute_lossx(4) + compute_lossx(3) + compute_lossx(2)
        return loss

    def inference(self, pred):
        """
        pred: (batch, 1)
        """
        out5 = torch.where(pred>self.t45, torch.tensor(5, dtype=torch.long, device=pred.device), torch.tensor(0, dtype=torch.long, device=pred.device))
        out4 = torch.where((pred>self.t34) & (pred<=self.t45), torch.tensor(4, dtype=torch.long, device=pred.device), torch.tensor(0, dtype=torch.long, device=pred.device))
        out3 = torch.where((pred>self.t23) & (pred<=self.t34), torch.tensor(3, dtype=torch.long, device=pred.device), torch.tensor(0, dtype=torch.long, device=pred.device))
        out2 = torch.where(pred<=self.t23, torch.tensor(2, dtype=torch.long, device=pred.device), torch.tensor(0, dtype=torch.long, device=pred.device))
        # pred_out = (out5 + out4 + out3 + out2).squeeze(dim=1)
        pred_out = (out5 + out4 + out3 + out2)
        return pred_out


if __name__ == '__main__':
    device = torch.device(2)
    udl = UnsureDataLoss(e_bool=False).to(device)
    label = torch.tensor([2, 2, 2,
                           3, 3, 3,
                           4, 4, 4,
                           5, 5, 5], device=device).reshape(12,1)
    pred = torch.tensor([-10, -5, 0,
                          -5, 0, 5,
                          0, 5, 10,
                          5, 10, 15], dtype=torch.float32 , device=device).reshape(12,1)
    print(udl(pred, label))
    print(udl.inference(pred))

    p23 = torch.sigmoid(-5-pred)
    p34 = torch.sigmoid(0-pred)
    p45 = torch.sigmoid(5-pred)

    bias = torch.tensor([1e-7], dtype=torch.float32, device=device, requires_grad=False)
    loss5 = -torch.sum(torch.log((1-p45)+bias)[9:12])
    loss4 = -torch.sum(torch.log((p45-p34)+bias)[6:9])
    loss3 = -torch.sum(torch.log((p34-p23)+bias)[3:6])
    loss2 = -torch.sum(torch.log((p23)+bias)[0:3])

    loss = (loss2 + loss3 + loss4 + loss5)/12
    print(loss2, loss3, loss4, loss5)
    print(loss)
