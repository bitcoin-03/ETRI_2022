import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32) # one_hot 형식으로 맞춤
        y_pred = F.softmax(y_pred, dim=1) # 합이 1이 되도록

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()

# https://github.com/Joonsun-Hwang/imbalance-loss-test/blob/main/Loss%20Test.ipynb
class LADELoss(nn.Module):
    def __init__(
        self, classes, img_max=512, prior=0.1, prior_txt=None, remine_lambda=0.1,
    ):
        super().__init__()
        if img_max is not None or prior_txt is not None:
            self.img_num_per_cls = (
                calculate_prior(classes, img_max, prior, prior_txt, return_num=True)
                .float()
                .cuda()
            )
            self.prior = self.img_num_per_cls / self.img_num_per_cls.sum()
        else:
            self.prior = None

        self.balanced_prior = torch.tensor(1.0 / classes).float().cuda()
        self.remine_lambda = remine_lambda

        self.num_classes = classes
        self.cls_weight = (
            self.img_num_per_cls.float() / torch.sum(self.img_num_per_cls.float())
        ).cuda()

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)

        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(
            x_p, x_q, num_samples_per_cls
        )
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, y_pred, target, q_pred=None):
        """
        y_pred: N x C
        target: N
        """
        per_cls_pred_spread = y_pred.T * (
            target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target)
        )  # C x N
        pred_spread = (
            y_pred
            - torch.log(self.prior + 1e-9)
            + torch.log(self.balanced_prior + 1e-9)
        ).T  # C x N

        num_samples_per_cls = torch.sum(
            target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1
        ).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(
            per_cls_pred_spread, pred_spread, num_samples_per_cls
        )

        loss = -torch.sum(estim_loss * self.cls_weight)
        return loss

def calculate_prior(num_classes, img_max=None, prior=None, prior_txt=None, reverse=False, return_num=False):
    if prior_txt:
        labels = []
        with open(prior_txt) as f:
            for line in f:
                labels.append(int(line.split()[1]))
        occur_dict = dict(Counter(labels))
        img_num_per_cls = [occur_dict[i] for i in range(num_classes)]
    else:
        img_num_per_cls = []
        for cls_idx in range(num_classes):
            if reverse:
                num = img_max * (prior ** ((num_classes - 1 - cls_idx) / (num_classes - 1.0)))
            else:
                num = img_max * (prior ** (cls_idx / (num_classes - 1.0)))
            img_num_per_cls.append(int(num))
    img_num_per_cls = torch.Tensor(img_num_per_cls)

    if return_num:
        return img_num_per_cls
    else:
        return img_num_per_cls / img_num_per_cls.sum()

# https://github.com/kaidic/LDAM-DRW
class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        print('합성곱',self.m_list[None, :].size(), index_float.transpose(0, 1).size())

        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1)) # 7 7 x 64
        batch_m = batch_m.view((-1, 1))

        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss,
    'f1': F1Loss,
    'focal': FocalLoss,
    'label_smoothing': LabelSmoothingLoss,
    "LADE": LADELoss,
    "LDAM" : LDAMLoss,
    "WCE" : nn.CrossEntropyLoss,
}


def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion

# loss
def loss_save(name, loss_data, with_clothes = False):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = {
        'daily': 7,
        'gender': 6,
        'embel': 3,
    }
    if with_clothes == True:
        criterion['clothes'] = 14
    if name == 'LDAM':
        beta = 0.9999
        for k,v in criterion.items():
            print('여기', len(loss_data.indices[f'{k}']),loss_data.indices[f'{k}'])
            train_sample = np.unique(loss_data.indices[f'{k}'],return_counts=True)[1] # 고유한 원소들만 모음, 배열이 2개가 담긴다. 각 원소들이 등장하는 횟수도 담김
            # indices -> numpy문법 index들이 담긴다.
            effective_num = 1.0 - np.power(beta, train_sample) # class num list 넣으면 되는데
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(train_sample)
            per_cls_weights = torch.FloatTensor(per_cls_weights).to(DEVICE)
            criterion[k] = create_criterion(name, cls_num_list=train_sample, max_m=0.5, s=30, weight=per_cls_weights).to(DEVICE)
    
    elif name == 'WCE': # weighted cross entropy
        daily_cnt = [465, 5580, 866, 1358, 448, 113, 351]
        gender_cnt = [116, 743, 2268, 2224, 3409, 421]
        embel_cnt = [4993, 2755, 1433]
        target_cnt = (daily_cnt,gender_cnt,embel_cnt)
        normed_weights = [0 for _ in range(len(criterion))]
        for i in range(len(criterion)):
            normed_weights[i] = [1 - (x / sum(target_cnt[i])) for x in target_cnt[i]]
            normed_weights[i] = torch.FloatTensor(normed_weights[i]).to(DEVICE)
        j = 0
        for k,v in criterion.items():
            criterion[k] = create_criterion(name, weight=normed_weights[j]).to(DEVICE)
            j+=1

    elif name in ['focal_2', 'focal', 'cross_entropy']:
        for k,v in criterion.items():
            criterion[k] = create_criterion(name).to(DEVICE)

    else:
        for k,v in criterion.items():
            criterion[k] = create_criterion(name, classes=v).to(DEVICE)
    return criterion