import os
import random
import time
import shutil
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics.ranking import roc_auc_score
import torchvision.transforms as transforms

from utils import Logger, AverageMeter, mkdir_p
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score

from utils.eval import mean_class_recall
from models.alexnet_neighbor_matching import AlexNet

"""
Configs
"""
parser = argparse.ArgumentParser(description='PyTorch Neighbor Matching Training')
# Optimization options
parser.add_argument('--epochs', default=1024, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[200, 500], 
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], 
                    help='LR is multiplied by gamma on schedule')

# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--howManyLabelled', type=int, default=300,
                        help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=250,
                        help='Number of labeled data')
parser.add_argument('--out', default='out',
                        help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)

#Add noise or data augmentation
parser.add_argument('--augu', action='store_true', default=False,
                    help='use augmentation or not!')
parser.add_argument('--noise', action='store_true', default=False,
                    help='use augmentation or not!')

#ManifoldMixup
parser.add_argument('--mixup', type=str, default = 'input')
parser.add_argument('--noSharp', action='store_true', default=False,
                    help='Avoid sharpeninig (for multilabel case!)')

#Dataset
parser.add_argument('--dataset', type=str, default = 'skin', choices =['xray', 'skin'])

#Supervised baseline
parser.add_argument('--sup', action='store_true', default=False,
                    help='supervised baseline!')

#Considering different pair for manifold mixup
parser.add_argument('--analyzeMN', default=False, action='store_true')
parser.add_argument('--bb', default=2, type=float)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)

if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
else:
    torch.manual_seed(args.manualSeed)

np.random.seed(args.manualSeed)

class GaussianNoise(nn.Module):
    def __init__(self, batch_size, input_shape, std=0.05, image_size=128):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = Variable(torch.zeros(self.shape).cuda())
        self.std = std
        self.image_size=image_size
    def forward(self, x):
        self.noise.data.normal_(0, std=self.std)
        try:
            return x + self.noise
        except:
            self.noise = Variable(torch.zeros((x.size(0),) + (1, self.image_size, self.image_size)).cuda())
            self.noise.data.normal_(0, std=self.std)
            return x + self.noise


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

def computeAUROC(dataGT, dataPRED, classCount):
    outAUROC = []
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    for i in range(classCount):
        try:
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
        except ValueError:
            pass
    return outAUROC

class SemiLossSum(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):

        # Lx = F.binary_cross_entropy(outputs_x, targets_x, reduction='sum')
        # Lu = F.mse_loss(outputs_u, targets_u, reduction='sum')
        # Lu = F.binary_cross_entropy(outputs_u, targets_u, reduction='sum')
        #Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        #Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1))
        Lx = -torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x)
        Lu = -torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)

class CELoss(object):
    def __call__(self, outputs_x, targets_x):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
    
        return Lx

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def main():

    print("Working for {}   alpha : {}  numOfLabelled : {}".format(args.mixup, args.alpha, args.howManyLabelled))

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    #Model and optimizer
    model = AlexNet(batch_size=args.batch_size, std=0.15, noise=args.noise, data=args.dataset)
    if use_cuda: model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    #Transforms for the data
    transformSequence = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor()
            ])
    
    trans_aug = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.RandomRotation(degrees=(-10,10)),
            transforms.RandomAffine(degrees=0,translate=(0.1,0.1)),
            transforms.ToTensor()
            ])

    from get_dataLoader_images import get_dataLoader_mix
    if args.augu:
        labeled_trainloader, unlabeled_trainloader, val_loader, test_loader = get_dataLoader_mix(
                transformSequence, trans_aug, labelled=args.howManyLabelled, batch_size=args.batch_size)
    else:
        labeled_trainloader, unlabeled_trainloader, val_loader, test_loader = get_dataLoader_mix(transformSequence,
                                                                                                    transformSequence,
                                                                                                    labelled=args.howManyLabelled,
                                                                                                    batch_size=args.batch_size)

    ntrain = len(labeled_trainloader.dataset)
    print('labeled trainloader length: ', ntrain)
    train_criterion = SemiLossSum()
    criterion=CELoss()
    start_epoch = 0
    best_AUC = 0

    # Resume
    title = 'latent-mixing'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_AUC = checkpoint['best_auc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(
            ['Train Loss', 'Train Loss X', 'Train Loss U', 'Valid Loss', 'Valid AUC', 'Valid ACC', 'Valid MCR', 'Test Loss', 'Test AUC', 'Test ACC', 'Test MCR'])

    writer = SummaryWriter(args.out)
    step = 0
    test_AUCS = []
    val_AUCS = []
    for epoch in range(start_epoch, args.epochs):

        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, current_learning_rate))
        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
                                                       train_criterion, epoch, use_cuda, args.mixup, args.noSharp)
        _, train_auc, _, _ = validate(labeled_trainloader, model, criterion, epoch, use_cuda, mode='Train Stats')
        val_loss, val_auc, val_acc, val_mcr = validate(val_loader, model, criterion, epoch, use_cuda, mode='Valid Stats')
        test_loss, test_auc, test_acc, test_mcr = validate(test_loader, model, criterion, epoch, use_cuda, mode='Test Stats ')

        step = args.val_iteration * (epoch + 1)

        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/valid_loss', val_loss, step)
        writer.add_scalar('losses/test_loss', test_loss, step)

        # writer.add_scalar('accuracy/train_acc', train_auc, step)
        writer.add_scalar('accuracy/val_auc', val_auc, step)
        writer.add_scalar('accuracy/test_auc', test_auc, step)
        writer.add_scalar('accuracy/test_acc', test_acc, step)
        writer.add_scalar('accuracy/test_mcr', test_mcr, step)
        

        # append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, val_loss, val_auc, val_acc, val_mcr, test_loss, test_auc, test_acc, test_mcr])

        # save model
        is_best = val_auc > best_AUC
        best_AUC = max(val_auc, best_AUC)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': val_auc,
                'best_auc': best_AUC,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        test_AUCS.append(test_auc)
        val_AUCS.append(val_auc)
    logger.close()
    writer.close()

    indx = np.argmax(val_AUCS)
    print('Best Val AUC: {} |    Best Test AUC (at best val): {}'.format(val_AUCS[indx], test_AUCS[indx]))
    print('Best Test AUC: {} |    Mean Test AUC: {}'.format(np.max(test_AUCS), np.mean(test_AUCS[-20:])))


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, criterion, epoch, use_cuda, mixup='input', noSharp=False, alr=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    # bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()

        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()

        with torch.no_grad():
            # update memory
            model(inputs_x, targets_x)
            # compute guessed labels of unlabel samples
            logits_u = model(inputs_u, is_matching=True)
            # targets_u = outputs_u.detach()
            p = logits_u
            '''
            _, _, outputs_u2 = model(inputs_u2, is_matching=True)
            p = (outputs_u + outputs_u2) / 2
            '''
            if not noSharp:
                pt = p**(1/args.T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
            else:
                targets_u = p
            targets_u = targets_u.detach()
            # print(targets_u.sum(dim=1))
            # print(targets_u)

        outputs_x = model(inputs_x)[1]
        outputs_u = model(inputs_u)[1]

        Lx, Lu, w = criterion(outputs_x, targets_x, outputs_u, targets_u, epoch+batch_idx/args.val_iteration)
        # w = args.lambda_u
        loss = Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    #     # uncomment this if you want to plot progress
    #     bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
    #                 batch=batch_idx + 1,
    #                 size=args.val_iteration,
    #                 data=data_time.avg,
    #                 bt=batch_time.avg,
    #                 total=bar.elapsed_td,
    #                 eta=bar.eta_td,
    #                 loss=losses.avg,
    #                 loss_x=losses_x.avg,
    #                 loss_u=losses_u.avg,
    #                 w=ws.avg
    #                 )
    #     bar.next()
    # bar.finish()
    return (losses.avg, losses_x.avg, losses_u.avg,)

def validate(valloader, model, criterion, epoch, use_cuda, mode):
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # bar = Bar(f'{mode}', max=len(valloader))
    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    total_val_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            # compute output
            _, outputs, logits = model(inputs)
            loss = criterion(outputs, targets)
            total_val_loss += loss.item()

            losses.update(loss.item(), inputs.size(0))

            outGT = torch.cat((outGT, targets.detach()), 0)
            outPRED = torch.cat((outPRED, logits.detach()), 0)

    aurocIndividual = computeAUROC(outGT, outPRED, 7)
    aurocMean = np.array(aurocIndividual).mean()
    
    outPRED = torch.argmax(outPRED, dim=1).cpu().data.numpy()
    outGT = torch.argmax(outGT, dim=1).cpu().data.numpy()
    acc = accuracy_score(outGT, outPRED)
    mcr = mean_class_recall(outGT, outPRED)

    return total_val_loss, aurocMean, acc, mcr

if __name__ == '__main__':
    main()