# -*- coding: utf-8 -*-

import torch


class MemoryPadding:
    def __init__(self, num_classes, dim, K):
        self.K = K  # memory size
        self.num_classes = num_classes
        self.dim = dim
        
        self.feats = torch.zeros(self.K, self.dim).cuda()
        self.targets = torch.zeros(self.K, self.num_classes).cuda()
        self.ptr = 0
        self.is_full = False

    def get(self):
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, targets):

        batch_size = feats.shape[0]

        q_size = int(self.ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.feats[q_size:q_size + batch_size] = feats
        self.targets[q_size:q_size + batch_size] = targets
        
        if (q_size + batch_size) >= self.K:
            self.is_full = True
        q_size = (q_size + batch_size) % self.K  # move pointer

        self.ptr = q_size
        


class MemoryPaddingClasswise:
    def __init__(self, num_classes, dim, K):
        self.K = K  # memory size per class
        self.num_classes = num_classes
        self.dim = dim
        
        self.feats = torch.zeros(self.num_classes, self.K, self.dim).cuda()
        self.targets = torch.zeros(self.num_classes, self.K, self.num_classes).cuda()
        self.ptr = [0]*self.num_classes
        self.count = [0]*self.num_classes
        
    def get(self):
        if min(self.count) >= self.K:
            feats = self.feats.reshape(self.num_classes*self.K, self.dim)
            targets = self.targets.reshape(self.num_classes*self.K, self.num_classes)
        else:
            feats = self.feats[:,:min(self.count)].reshape(self.num_classes*min(self.count), self.dim)
            targets = self.targets[:,:min(self.count)].reshape(self.num_classes*min(self.count), self.num_classes)
        return feats, targets

    def enqueue_dequeue(self, feats, targets):       
        for i in range(self.num_classes):
            class_i = targets[:,i]      
            cur_feats = feats[class_i.eq(1)]
            cur_targets = torch.zeros(targets.shape).cuda()
            cur_targets[:,i] = 1
            cur_targets = cur_targets[class_i.eq(1)]
            # print(cur)
            q_size = len(cur_feats)

            if self.ptr[i] + q_size > self.K:
                if q_size >= self.K:
                    self.feats[i,:] = cur_feats[:self.K]
                    self.targets[i,:] = cur_targets[:self.K]
                    self.ptr[i] = 0
                else:
                    t = (self.ptr[i] + q_size) % self.K
                    if q_size-t > 0:
                        self.feats[i, t-q_size:] = cur_feats[:q_size-t]
                        self.targets[i, t-q_size:] = cur_targets[:q_size-t]
                    self.feats[i, :t] = cur_feats[-t:]
                    self.targets[i, :t] = cur_targets[-t:]
                    self.ptr[i] = t
            else:
                self.feats[i, self.ptr[i]: self.ptr[i] + q_size] = cur_feats
                self.targets[i, self.ptr[i]: self.ptr[i] + q_size] = cur_targets
                self.ptr[i] += q_size
            self.count[i] += q_size
            
            

class SafeMemoryPaddingClasswise:
    def __init__(self, num_classes, dim, K):
        self.K = K  # memory size per class
        self.num_classes = num_classes
        self.dim = dim
        
        self.feats = torch.zeros(self.num_classes, self.K, self.dim).cuda()
        self.targets = torch.zeros(self.num_classes, self.K, self.num_classes).cuda()
        self.ptr = [0]*self.num_classes
        self.count = [0]*self.num_classes
        
    def get(self):
        if min(self.count) >= self.K:
            feats = self.feats.reshape(self.num_classes*self.K, self.dim)
            targets = self.targets.reshape(self.num_classes*self.K, self.num_classes)
        else:
            feats = self.feats[:,:min(self.count)].reshape(self.num_classes*min(self.count), self.dim)
            targets = self.targets[:,:min(self.count)].reshape(self.num_classes*min(self.count), self.num_classes)
        return feats, targets

    def enqueue_dequeue(self, feats, targets, preds):
        n, d = preds.shape
        preds_1 = (preds >= 0.3)
        preds_2 = torch.argmax(preds, 1).reshape(n, 1)
        preds_2 = torch.zeros(n, self.num_classes).cuda().scatter_(1, preds_2, 1)
        preds = (preds_1 | preds_2.type_as(preds_1)).type_as(targets)
        # preds = torch.argmax(preds, 1).reshape(n, 1)
        # preds = torch.zeros(n, self.num_classes).cuda().scatter_(1, preds, 1)
        torch.set_printoptions(edgeitems=14)
        # print(preds)
        for i in range(self.num_classes):
            # class_i = targets[:,i]
            class_i = ((targets[:, i] + preds[:, i] > 1) .type_as(targets))
            # print('target: ', targets[:, i])
            # print('pred: ', preds[:, i])
            cur_feats = feats[class_i.eq(1)]
            cur_targets = torch.zeros(targets.shape).cuda()
            cur_targets[:,i] = 1
            cur_targets = cur_targets[class_i.eq(1)]
            # print(cur_targets.shape)
            q_size = len(cur_feats)

            if self.ptr[i] + q_size > self.K:
                if q_size >= self.K:
                    self.feats[i,:] = cur_feats[:self.K]
                    self.targets[i,:] = cur_targets[:self.K]
                    self.ptr[i] = 0
                else:
                    t = (self.ptr[i] + q_size) % self.K
                    if q_size-t > 0:
                        self.feats[i, t-q_size:] = cur_feats[:q_size-t]
                        self.targets[i, t-q_size:] = cur_targets[:q_size-t]
                    self.feats[i, :t] = cur_feats[-t:]
                    self.targets[i, :t] = cur_targets[-t:]
                    self.ptr[i] = t
            else:
                self.feats[i, self.ptr[i]: self.ptr[i] + q_size] = cur_feats
                self.targets[i, self.ptr[i]: self.ptr[i] + q_size] = cur_targets
                self.ptr[i] += q_size
            self.count[i] += q_size
