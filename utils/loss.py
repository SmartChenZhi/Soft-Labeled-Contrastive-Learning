import torch
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn

import math
import numpy as np


def jaccard_loss(true, logits, eps=1e-7):
    """
    Computes the Jaccard loss, a.k.a the IoU loss.
    Args:
        true: the ground truth of shape [B, H, W] or [B, 1, H, W]
        logits: the output of the segmentation model (without softmax) [B, C, H, W]
        eps:

    Returns:
    The Jaccard loss
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1, device=true.device)[true.squeeze(1)]
        true_1_hot = torch.moveaxis(true_1_hot, -1, 1)
        true_1_hot_f = true_1_hot[:, 0:1, :, :]  # background
        true_1_hot_s = true_1_hot[:, 1:2, :, :]  # foreground
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes, device=true.device)[true.squeeze(1)]
        true_1_hot = torch.moveaxis(true_1_hot, -1, 1)  # B, C, H, W
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true_1_hot.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return 1 - jacc_loss


def loss_calc(pred, label, gpu=0, jaccard=False):
    """
    This function returns cross entropy loss plus jaccard loss for semantic segmentation
    Args:
        pred: the logits of the prediction with shape [B, C, H, W]
        label: the ground truth with shape [B, H, W]
        gpu: the gpu number
        jaccard: if apply jaccard loss

    Returns:

    """
    label = Variable(label.long()).to(gpu)
    criterion = nn.CrossEntropyLoss().to(gpu)
    loss = criterion(pred, label)
    if jaccard:
        loss = loss + jaccard_loss(true=label, logits=pred)
    return loss


def dice_loss(pred, target):
    """
    input is a torch variable of size [N,C,H,W]
    target: [N,H,W]
    """
    n, c, h, w = pred.size()
    pred = pred.cuda()
    target = target.cuda()
    target_onehot = torch.zeros([n, c, h, w]).cuda()
    target = torch.unsqueeze(target, dim=1)  # n*1*h*w
    target_onehot.scatter_(1, target.long(), 1)

    assert pred.size() == target_onehot.size(), "Input sizes must be equal."
    assert pred.dim() == 4, "Input must be a 4D Tensor."
    uniques = np.unique(target_onehot.cpu().data.numpy())
    assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    eps = 1e-5
    probs = F.softmax(pred, dim=1)
    num = probs * target_onehot  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)  # b,c,h
    num = torch.sum(num, dim=2)  # b,c,

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)  # b,c,h
    den1 = torch.sum(den1, dim=2)  # b,c,

    den2 = target_onehot * target_onehot  # --g^2
    den2 = torch.sum(den2, dim=3)  # b,c,h
    den2 = torch.sum(den2, dim=2)  # b,c

    dice = 2.0 * (num / (den1 + den2 + eps))  # b,c

    dice_total = torch.sum(dice) / dice.size(0)  # divide by batch_sz
    return 1 - dice_total / c


def loss_entropy(pred, device, smooth, mode='mean'):
    assert pred.ndim == 4
    assert mode == 'mean' or mode == 'sum'
    C = pred.size()[1]
    ent = pred * torch.log(pred + smooth)
    loss = (-1 / torch.log(torch.tensor(C).to(device))) * ent.sum(dim=1)
    if mode == 'mean':
        loss = loss.mean()
    elif mode == 'sum':
        loss = loss.sum(dim=tuple(np.arange(loss.ndim)[1:])).mean()
    else:
        raise NotImplementedError
    return loss


def loss_entropy_BCL(p):
    """
    entropy loss used in BCL
    :param p:
    :return:
    """
    p = F.softmax(p, dim=1)
    log_p = F.log_softmax(p, dim=1)
    loss = -torch.sum(p * log_p, dim=1)
    return loss


def cosine_similarity_BCL(class_list, label_resize, feature, label2, feature2, num_class=19):
    """
    calculate the cosine similarity between centroid of one domain and individual features of another domain
    @param class_list: the unique class index of the label
    @param label_resize: (1, 1, h, w) the label of the first domain
    @param feature: (1, C, h, w) the feature of the first domain
    @param label2: (1, 1, H, W) the label of the second domain (full size)
    @param feature2: (1, C, h, w) the feature of the second domain
    @param num_class: the total number of classes
    @return:
    """
    # get the shape of the feature
    _, ch, feature_h, feature_w = feature.size()
    prototypes = torch.zeros(size=(num_class, ch)).cuda()
    for i, index in enumerate(class_list):
        # enumerate over the class index in the class_list, class 255 is ignored
        if index != 255.:
            fg_mask = ((label_resize == index) * 1).cuda().detach()  # extract the mask for label == index
            # mask out the features correspond to certain class index and calculate the masked average feature
            prototype = (fg_mask * feature).squeeze().reshape(ch, feature_h * feature_w).sum(
                -1) / fg_mask.sum()
            prototypes[int(index)] = prototype  # (class_num, ch) register the prototypes into the list

    # (class_num, feature_h * feature_w) the cosine similarity between each class in one domain and each
    # individual feature in another domain
    cs_map = torch.matmul(F.normalize(prototypes, dim=1),
                          F.normalize(feature2.squeeze().reshape(ch, feature_h * feature_w), dim=0))
    # set the value to -1 (smallest in cosine value) when the class index does not overlap in both two domain
    cs_map[cs_map == 0] = -1
    # make sure that label and label2 have the same shape
    cosine_similarity_map = F.interpolate(cs_map.reshape(1, num_class, feature_h, feature_w), size=label2.size()[-2:])
    cosine_similarity_map *= 10
    return cosine_similarity_map


def bidirect_contrastive_loss_BCL(feature_s, label_s, feature_t, label_t, num_class, config):
    """
    Calculate the contrastive loss between two features of different domains
    @param feature_s:  1, C, h, w the source feature
    @param label_s: 1, H, W the source (pseudo)label
    @param feature_t: 1, C, h, w the target feature
    @param label_t: 1, H, W the target (pseudo)label
    @param num_class: the total number of classes
    @param config: the configuration variable
    @return:
    """

    # interpolate(down-sample) the labels to have the same size as the features
    _, ch, feature_s_h, feature_s_w = feature_s.size()
    label_s_resize = F.interpolate(label_s.float().unsqueeze(0), size=(feature_s_h, feature_s_w))  # (1, 1, h, w)
    _, _, feature_t_h, feature_t_w = feature_t.size()
    label_t_resize = F.interpolate(label_t.float().unsqueeze(0), size=(feature_t_h, feature_t_w))  # (1, 1, h, w)

    # get the unique class number for both source and target (pseudo)labels
    source_list = torch.unique(label_s_resize.float())
    target_list = torch.unique(label_t_resize.float())

    # find the overlapping class index except 255
    overlap_classes = [int(index.detach()) for index in source_list if index in target_list and index != 255]
    # calculate the similarity map
    cosine_similarity_map = cosine_similarity_BCL(source_list, label_s_resize, feature_s, label_t, feature_t, num_class)

    cross_entropy_weight = torch.zeros(num_class, dtype=torch.float, device=feature_s.device)
    cross_entropy_weight[overlap_classes] = 1.0
    prototype_loss = torch.nn.CrossEntropyLoss(weight=cross_entropy_weight, ignore_index=255)

    # generate the prediction map containing class indices where uncertainty pixels are set to 255
    prediction_by_cs = F.softmax(cosine_similarity_map, dim=1)  # compute the softmax of the similarity map
    target_predicted = prediction_by_cs.argmax(dim=1)  #
    confidence_of_target_predicted = prediction_by_cs.max(dim=1).values  # the max value for each category
    masked_target_predicted = torch.where(confidence_of_target_predicted > .8, target_predicted, 255)
    masked_target_predicted_resize = F.interpolate(masked_target_predicted.float().unsqueeze(0),
                                                   size=(feature_t_h, feature_t_w), mode='nearest')
    # set the pixels to 255 if the pixel in the (pseudo)label is 255 (uncertainty)
    label_t_resize_new = label_t_resize.clone().contiguous()
    label_t_resize_new[label_t_resize_new == 255] = masked_target_predicted_resize[label_t_resize_new == 255]
    target_list2 = torch.unique(label_t_resize_new)

    cosine_similarity_map2 = cosine_similarity_BCL(target_list2, label_t_resize_new, feature_t, label_s, feature_s, num_class)

    metric_loss1 = prototype_loss(cosine_similarity_map, label_t)
    metric_loss2 = prototype_loss(cosine_similarity_map2, label_s)

    metric_loss = config.lamb_metric1 * metric_loss1 + config.lamb_metric2 * metric_loss2
    return metric_loss


def loss_class_prior(pred, prior, w, device):
    prob_pred = pred.mean(dim=(0, 2, 3))
    loss = torch.nn.ReLU()(w * prior - prob_pred)
    return loss.sum()


def exp_func(v1, v2, tau=5):
    h = torch.exp((torch.matmul(v1, v2) / tau))
    return h


class ContrastiveLoss(nn.Module):
    def __init__(self, tau=5, n_class=4, bg=False, norm=True):
        super(ContrastiveLoss, self).__init__()
        self._tau = tau
        # self._n_class = n_class
        # self._bg = bg
        self._norm = norm

    def forward(self, centroid_s, centroid_t, bg=False, split=False):  # centroid_s (4, 32)
        norm_t = torch.norm(centroid_t, p=2, dim=1, keepdim=True)  # (4, 1)
        if self._norm:
            norm_s = torch.norm(centroid_s, p=2, dim=1, keepdim=True)  # (4, 1) compute the L2 norm of each centroid
            centroid_s = centroid_s / (norm_s + 1e-7)
            centroid_t = centroid_t / (norm_t + 1e-7)
        # a matrix with shape (#class, 2 * #class) storing the exponential values between two centroids
        # centroid_matrix = torch.zeros(n_class, 2 * n_class)
        # n_class = centroid_s.size()[0]
        # loss = 0
        # for i in range(0 if bg else 1, n_class):
        #     exp_sum = 0
        #     exp_self = 0
        #     for j in range(n_class):
        #         if i == j:
        #             exp_self = exp_func(centroid_t[i], centroid_s[j], tau=self._tau) + \
        #                        exp_func(centroid_t[i], centroid_t[j], tau=self._tau)
        #             exp_sum = exp_sum + exp_self
        #         else:
        #             exp_sum = exp_sum + exp_func(centroid_t[i], centroid_s[j], tau=self._tau) + \
        #                       exp_func(centroid_t[i], centroid_t[j], tau=self._tau)
        #     logit = -torch.log(exp_self / (exp_sum + 1e-7))
        #     loss = loss + logit
        exp_mm = torch.exp(torch.mm(centroid_t, centroid_s.transpose(0, 1)))
        exp_mm_t = torch.exp(torch.mm(centroid_t, centroid_t.transpose(0, 1)))
        diag_idx = torch.arange(0 if bg else 1, 4, dtype=torch.long)
        denom = exp_mm[0 if bg else 1:].sum(dim=1) + exp_mm_t[0 if bg else 1:].sum(dim=1)
        if split:
            nom1, nom2 = exp_mm[diag_idx, diag_idx], exp_mm_t[diag_idx, diag_idx]
            logit = 0.5 * (-torch.log(nom1 / (denom + 1e-7)) - torch.log(nom2 / (denom + 1e-7)))
        else:
            nom = exp_mm[diag_idx, diag_idx] + exp_mm_t[diag_idx, diag_idx]
            logit = -torch.log(nom / (denom + 1e-7))
        loss = logit.sum()
        return loss


def contrastive_loss(centroid_s, centroid_t, tau=5, n_class=4, bg_included=False, norm=False):
    """

    :param centroid_s: (4, 32)
    :param centroid_t:
    :param tau: temperature parameter
    :param n_class: the number of classes in the label
    :param bg_included: if the background is included in the computation of contrastive loss
    :param norm: whether divide norms in the contrastive loss
    :return:
    """
    if norm:
        norm_s = torch.norm(centroid_s, p=2, dim=1, keepdim=True)  # (4, 1) compute the L2 norm of each centroid
        norm_t = torch.norm(centroid_t, p=2, dim=1, keepdim=True)  # (4, 1)
        centroid_s = centroid_s / norm_s
        centroid_t = centroid_t / norm_t
    # a matrix with shape (#class, 2 * #class) storing the exponential values between two centroids
    # centroid_matrix = torch.zeros(n_class, 2 * n_class)
    loss = 0
    for i in range(0 if bg_included else 1, n_class):
        exp_sum = 0
        exp_self = 0
        for j in range(n_class):
            if j != i:
                exp_sum = exp_sum + exp_func(centroid_t[i], centroid_t[j], tau=tau)
        for j in range(n_class):
            if i == j:
                exp_self = exp_func(centroid_t[i], centroid_s[j], tau=tau)
            exp_sum = exp_sum + exp_func(centroid_t[i], centroid_s[j], tau=tau)
        logit = torch.unsqueeze(torch.unsqueeze(torch.log(torch.div(exp_self, exp_sum)), 0), 0)
        loss_class = nn.NLLLoss()(logit, torch.tensor([0], requires_grad=False).cuda())
        if math.isnan(loss_class.item()):
            print('nan!!!')
        loss = loss + loss_class
    return loss


class SupConLoss(nn.Module):
    """modified supcon loss for segmentation application, the main difference is that the label for different view
    could be different if after spatial transformation"""

    def __init__(self, temperature=0.07,
                 contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None):
        # input features shape: [bsz, v, c, w, h]
        # input labels shape: [bsz, v, w, h]
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if features.ndim <= 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 4 dimensions are required')
        if features.ndim == 5:
            contrast_count = features.shape[1]
            contrast_feature = torch.cat(torch.unbind(features, dim=1),
                                         dim=0)  # of size (bsz*v, c, h, w) (2, 32, 56, 56)

        kernels = contrast_feature.permute(0, 2, 3, 1)  # (2, 56, 56, 32)
        kernels = kernels.reshape(-1, contrast_feature.shape[1], 1, 1)  # (6272, 32, 1, 1)
        # kernels = kernels[non_background_idx]
        logits = torch.div(F.conv2d(contrast_feature, kernels),
                           self.temperature)  # of size (bsz*v, bsz*v*h*w, h, w) (2, 6272, 56, 56)
        logits = logits.permute(1, 0, 2, 3)  # (6272, 2, 56, 56)
        logits = logits.reshape(logits.shape[0],
                                -1)  # (6272, 6272) the vector multiplication of the combination of every two vector features

        if labels is not None:
            labels = torch.cat(torch.unbind(labels, dim=1), dim=0)
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)

            bg_bool = torch.eq(labels.squeeze().cpu(), torch.zeros(labels.squeeze().shape))
            non_bg_bool = ~ bg_bool
            non_bg_bool = non_bg_bool.int().to(device)
        else:
            mask = torch.eye(logits.shape[0] // contrast_count).float().to(device)  # (3136, 3136)
            mask = mask.repeat(contrast_count, contrast_count)  # (6272, 6272)
            # print(mask.shape)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(mask.shape[0]).view(-1, 1).to(device), 0
        )  # (6272, 6272) replace the diagonal of the ones matrix with 0.
        mask = mask * logits_mask  # mask[:3136, 3136: 6272] and mask[3136: 6272, : 3136] are diagonal matrices. Other part of mask is 0.

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # Reduce diagonal to 0 and keep other elements unchanged
        log_prob = logits - torch.log(
            exp_logits.sum(1, keepdim=True))  # (6272, 6272) log(exp(v1*v2)) - log(sum(exp(vi*vj))) = Contrasitve loss

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(
            1)  # (6272,) only include the elements that represent the positive pair with the row sample

        # loss
        loss = - mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        if labels is not None:
            # only consider the contrastive loss for non-background pixel
            loss = (loss * non_bg_bool).sum() / (non_bg_bool.sum())
        else:
            loss = loss.mean()
        return loss


class LocalConLoss(nn.Module):
    def __init__(self, temperature=0.7, stride=4):
        super(LocalConLoss, self).__init__()
        self.temp = temperature
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.supconloss = SupConLoss(temperature=self.temp)
        self.stride = stride

    def forward(self, features, labels=None):
        # input features: [bsz, num_view, c, h ,w], h & w are the image size
        # resample feature maps to reduce memory consumption and running time
        features = features[:, :, :, ::self.stride, ::self.stride]

        if labels is not None:
            labels = labels[:, :, ::self.stride, ::self.stride]
            if labels.sum() == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss

            loss = self.supconloss(features, labels)
            return loss
        else:
            loss = self.supconloss(features)
            return loss


class BlockConLoss(nn.Module):
    def __init__(self, temperature=0.7, block_size=32):
        super(BlockConLoss, self).__init__()
        self.block_size = block_size
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.supconloss = SupConLoss(temperature=temperature)

    def forward(self, features, labels=None):
        # input features: [bsz, num_view, c, h ,w], h & w are the image size
        shape = features.shape  # (1, 2, 32, 224, 224)
        img_size = shape[-1]
        div_num = img_size // self.block_size  # 14
        if labels is not None:
            loss = []
            for i in range(div_num):
                # print("Iteration index:", idx, "Batch_size:", b)
                for j in range(div_num):
                    # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                    block_features = features[:, :, :, i * self.block_size:(i + 1) * self.block_size,
                                     j * self.block_size:(j + 1) * self.block_size]
                    block_labels = labels[:, :, i * self.block_size:(i + 1) * self.block_size,
                                   j * self.block_size:(j + 1) * self.block_size]

                    if block_labels.sum() == 0:
                        continue

                    tmp_loss = self.supconloss(block_features, block_labels)
                    loss.append(tmp_loss)

            if len(loss) == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss
            loss = torch.stack(loss).mean()
            return loss

        else:
            loss = []
            for i in range(div_num):
                # print("Iteration index:", idx, "Batch_size:", b)
                for j in range(div_num):
                    # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                    block_features = features[:, :, :, i * self.block_size:(i + 1) * self.block_size,
                                     j * self.block_size:(
                                                                     j + 1) * self.block_size]  # (1, 2, 32, block_size, block_size)

                    tmp_loss = self.supconloss(block_features)

                    loss.append(tmp_loss)

            loss = torch.stack(loss).mean()  # torch.stack(loss).size() = 196
            return loss


class MPCL(nn.Module):
    def __init__(self, device, num_class=5, temperature=0.07, m=0.5,
                 base_temperature=0.07, easy_margin=False):
        super(MPCL, self).__init__()
        self.num_class = num_class
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  # -0.8775825618903726
        self.mm = math.sin(math.pi - m) * m  # 0.23971276930210156
        self.device = device
        self.easy_margin = easy_margin

    def forward(self, features, labels, class_center_feas,
                pixel_sel_loc=None, mask=None):
        """
        :param features: [B * H * W] * 1 * C  normalized.
        :param labels: B * H * W.
        :param class_center_feas: class prototypes C * #class.
        :param pixel_sel_loc: mask to select features for the loss, [B * H * W].
        :param mask: mask that indicate which class the pixel belongs to.
        :return: the pixel-wise contrastive loss
        """
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        """build mask"""
        num_samples = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(num_samples, dtype=torch.float32).cuda()
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1).long()  # n_sample*1
            class_center_labels = torch.arange(0, self.num_class).long().cuda()
            # print(class_center_labels)
            class_center_labels = class_center_labels.contiguous().view(-1, 1)  # n_class*1
            if labels.shape[0] != num_samples:
                raise ValueError('Num of labels does not match num of features')
            """convert to one-hot encoded mask [B * H * W] * #class indicating the positive class of each pixel"""
            mask = torch.eq(labels,
                            torch.transpose(class_center_labels, 0, 1)).float().cuda()  # broadcast n_sample*n_class
        else:
            mask = mask.float().cuda()
        # n_sample = batch_size * fea_h * fea_w
        # mask n_sample*n_class  the mask_ij represents whether the i-th sample has the same label with j-th class or not.
        # in our experiment, the n_view = 1, so the contrast_count = 1
        contrast_count = features.shape[1]  # 1
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [n*h*w]*fea_s

        anchor_feature = contrast_feature  # [B * H * W] * C
        anchor_count = contrast_count  # 1

        """compute logits"""
        # dot product between the individual features and the class centroids, cos(a)
        cosine = torch.matmul(anchor_feature, class_center_feas)  # [n*h*w] * n_class
        logits = torch.div(cosine, self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0.0001, 1.0))
        """cos(a + m)"""
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # print(phi)
        phi_logits = torch.div(phi, self.temperature)

        phi_logits_max, _ = torch.max(phi_logits, dim=1, keepdim=True)
        phi_logits = phi_logits - phi_logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)

        tag_1 = (1 - mask)
        tag_2 = mask
        """the elements of the denominator"""
        exp_logits = torch.exp(logits * tag_1 + phi_logits * tag_2)  # [B * H * W] * #class
        phi_logits = (logits * tag_1) + (phi_logits * tag_2)  # [B * H * W] * #class
        """log(exp(phi_logits) / exp_logits.sum(1)). contrastive loss for each pixel."""
        log_prob = phi_logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-4)  # [B * H * W] * #class

        if pixel_sel_loc is not None:

            pixel_sel_loc = pixel_sel_loc.view(-1)

            mean_log_prob_pos = (mask * log_prob).sum(1)
            mean_log_prob_pos = pixel_sel_loc * mean_log_prob_pos
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = torch.div(loss.sum(), pixel_sel_loc.sum() + 1e-4)
        else:

            mean_log_prob_pos = (mask * log_prob).sum(1)  # [B * H * W]
            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.view(anchor_count, num_samples).mean()

        return loss


def mpcl_loss_calc(feas, labels, class_center_feas, loss_func,
                   pixel_sel_loc=None, tag='source'):
    '''
    feas:  batch*c*h*w
    label: batch*img_h*img_w
    class_center_feas: n_class*n_feas
    '''

    n, c, fea_h, fea_w = feas.size()
    if tag == 'source' and (labels.size()[1] != fea_h or labels.size()[2] != fea_w):
        labels = labels.float()
        labels = F.interpolate(labels, size=fea_w, mode='nearest')
        labels = labels.permute(0, 2, 1).contiguous()
        labels = F.interpolate(labels, size=fea_h, mode='nearest')
        labels = labels.permute(0, 2, 1).contiguous()  # batch*fea_h*fea_w

    labels = labels.cuda()
    labels = labels.view(-1).long()

    feas = torch.nn.functional.normalize(feas, p=2, dim=1)
    feas = feas.transpose(1, 2).transpose(2, 3).contiguous()  # batch*c*h*w->batch*h*c*w->batch*h*w*c
    feas = torch.reshape(feas, [n * fea_h * fea_w, c])  # [batch*h*w] * c
    feas = feas.unsqueeze(1)  # [batch*h*w] 1 * c

    class_center_feas = torch.nn.functional.normalize(class_center_feas, p=2, dim=1)
    class_center_feas = torch.transpose(class_center_feas, 0, 1)  # n_fea*n_class

    loss = loss_func(feas, labels, class_center_feas,
                     pixel_sel_loc=pixel_sel_loc)
    return loss


def batch_pairwise_dist(x, y):
    # N, 2500, 3 | 8, 300, 3
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))  # x^2 (N * 300 * 300)
    yy = torch.bmm(y, y.transpose(2, 1))  # y^2
    zz = torch.bmm(x, y.transpose(2, 1))  # xy
    diag_ind = torch.arange(0, num_points).long().cuda()
    rx = xx[:, diag_ind, diag_ind]  # (N, 300)
    rx = rx.unsqueeze(1)  # (N, 1, 300)
    rx = rx.expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz).clip(min=0)  # (x - y)^2 = x^2 + y^2 - 2xy
    return P


def batch_NN_loss(x, y):
    smooth = 1e-7
    bs, num_points, points_dim = x.size()
    dist1 = torch.sqrt(batch_pairwise_dist(x, y) + smooth)  # (N, 300, 300)
    values1, indices1 = dist1.min(dim=2)  # (N, 300)

    dist2 = torch.sqrt(batch_pairwise_dist(y, x) + smooth)
    values2, indices2 = dist2.min(dim=2)
    a = torch.div(torch.sum(values1, 1), num_points)
    b = torch.div(torch.sum(values2, 1), num_points)
    sum = torch.div(torch.sum(a), bs) + torch.div(torch.sum(b), bs)

    return sum
