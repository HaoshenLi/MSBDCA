import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, opts, ignore_index=-1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, batch):
        loss = self.criterion(inputs, targets)
        return {"loss": loss}
    

class CrossEntropyLoss_multitask(nn.Module):
    def __init__(self, opts, ignore_index=-1):
        super().__init__()
        self.criterion_clf = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.criterion_sgt = nn.CrossEntropyLoss()

    def forward(self, inputs, targets, mask):
        out_sgt, out_clf = inputs
        mask = mask.long()
        mask = mask[:, 1][:, 0]
        loss_sgt = self.criterion_sgt(out_sgt, mask)
        loss_clf = self.criterion_clf(out_clf, targets)
        return {"loss_sgt": loss_sgt,
                "loss_clf": loss_clf,
                "loss": loss_sgt + loss_clf}
    

class DeepSurvLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _compute_loss(self, P, T, E, M, mode):  #P_risk, T, E, M_risk
        P_exp = torch.exp(P) # (B,)
        P_exp_B = torch.stack([P_exp for _ in range(P.shape[0])], dim=0) # (B, B)
        if mode == 'risk':
            E = E.float() * (M.sum(dim=1) > 0).float()
        elif mode == 'surv':
            E = (M.sum(dim=1) > 0).float()
        else:
            raise NotImplementedError
        P_exp_sum = (P_exp_B * M.float()).sum(dim=1)
        loss = -torch.sum(torch.log(P_exp / (P_exp_sum+1e-6)) * E) / torch.sum(E)
        return loss

    def forward(self, P_risk, P_surv, T, E):
        # P: (B,)
        # T: (B,)
        # E: (B,) \in {0, 1}
        M_risk = T.unsqueeze(dim=1) <= T.unsqueeze(dim=0) # (B, B)
        M_surv = T.unsqueeze(dim=1) > T.unsqueeze(dim=0) # (B, B)
        M_surv = M_surv.float() * torch.stack([E for _ in range(P_surv.shape[0])], dim=0).float()
        loss_risk = self._compute_loss(P_risk, T, E, M_risk, mode='risk')
        loss_surv = self._compute_loss(P_surv, T, E, M_surv, mode='surv')
        return loss_risk, loss_surv
    

class DSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ds_criterion = DeepSurvLoss()
    
    def forward(self, pred, label, batch):
        # pred: (B, 2)
        # label: (B,)
        
        OS, OSCensor = batch["OS"].float().to(pred.device), batch["OSCensor"].float().to(pred.device)
        surv_pred = torch.softmax(pred, dim=1)
        risk_loss, _ = self.ds_criterion(surv_pred[:, 0], surv_pred[:, 1], OS, OSCensor)
        return {
            "loss": risk_loss
        }


class DSCELoss(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.ds_criterion = DeepSurvLoss()
        self.ce_weight = opts.ce_weight
        self.ds_weight = opts.ds_weight
        
    def forward(self, pred, label, batch):
        # pred: (B, 2)
        # label: (B,)
        ce_loss = self.ce_criterion(pred, label.long())
        
        OS, OSCensor = batch["OS"].float().to(pred.device), batch["OSCensor"].float().to(pred.device)
        surv_pred = torch.softmax(pred, dim=1)
        risk_loss, _ = self.ds_criterion(surv_pred[:, 0], surv_pred[:, 1], OS, OSCensor)
        
        loss = ce_loss * self.ce_weight + risk_loss * self.ds_weight
        
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "risk_loss": risk_loss,
        }


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, ignore_index=-1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Forward pass of the Focal Loss.
        
        Args:
            inputs (torch.Tensor): Input predictions, shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels, shape (batch_size,).
            
        Returns:
            torch.Tensor: Loss value.
        """
        # Ignore -1 labels
        valid_indices = targets != self.ignore_index
        targets = targets[valid_indices]
        inputs = inputs[valid_indices]

        # Compute focal loss
        probs = torch.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1))
        at = (self.alpha * (1 - targets) + (1 - self.alpha) * targets).unsqueeze(1)
        focal_weights = (1 - pt) ** self.gamma
        weighted_ce_loss = at * focal_weights * torch.log(pt + 1e-10)
        focal_loss = torch.sum(-weighted_ce_loss)

        # Normalize the loss
        num_valid = torch.sum(valid_indices)
        if num_valid > 0:
            focal_loss /= num_valid

        return focal_loss


class LabelSmoothing(nn.Module):
    #NLL loss with label smoothing.

    def __init__(self, smoothing=0.1):
        #具体实现:(1 - epsilon) * crossentropy_loss + epsilon / K * sum(log(y_pred))
        #与公式有出入
        
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()






