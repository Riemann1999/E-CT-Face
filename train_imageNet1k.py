import argparse
import torch
import os
from torchvision import datasets, transforms
from layer.WarmupCosineLR import WarmupCosineLR
from layer.LabelSmoothingCrossEntropy import LabelSmoothing
import torch.optim as optim

from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
from data import cfg_mv2, cfg_mv1, cfg_mv3, cfg_m3_075, cfg_efficient, cfg_MVN

from model.Ctdn import Ctdn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


# def get_model_state_dict(model):
#     if isinstance(model, EMA):
#         return get_model_state_dict(model.ema_model)
#     else:
#         return model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def val(model, val_loader, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_id, (images, target) in enumerate(val_loader):
            images = Variable(images).cuda()
            target = Variable(target).cuda()
            # compute output
            output = model(images)
            # loss = criterion(output, target)
            loss = criterion(images, output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(' Acc Top1: {top1.avg:.3f} ||  Top5: {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg, losses.avg


def main(args):
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_dir = args.data_train_dir
    val_dir = args.data_val_dir
    batch_size = args.batch_size
    log_dir = args.log_dir
    EPOCH = args.epochs
    weights = args.weights
    decay1 = args.decay1
    decay2 = args.decay2
    max_lr = args.lr
    min_lr = args.init_lr
    weight_decay = args.weight_decay
    cfg = None
    if args.network == "mobilevit2_1.00":
        cfg = cfg_mv1
    elif args.network == "mobilevit2_0.75":
        cfg = cfg_mv2
    elif args.network == "mobilevitv2_200_384_in22ft1k":
        cfg = cfg_mv3
    elif args.network == "mobilenetv3_075":
        cfg = cfg_m3_075
    elif args.network == "efficientNet":
        cfg = cfg_efficient
    elif args.network == "mobilevit2_new":
        cfg = cfg_MVN
    if os.path.exists("./weights_Cls") is False:
        os.makedirs("./weights_Cls")

    
    train_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),  
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),  
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=16,
                                                   pin_memory=True)  # ,num_workers=16,pin_memory=False
    val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=16,
                                                 pin_memory=True)  # ,num_workers=16,pin_memory=True
    # model = create_model(num_classes = num_classes)

    model = Ctdn(cfg=cfg, phase='train', label='cls').to('cuda')
    print(model)

    model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay, amsgrad=False)
    # lr_she = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0= 500,T_mult=2)
    lr_she = WarmupCosineLR(optimizer, lr_min=min_lr, lr_max=max_lr, warm_up=3 * len(train_dataloader),
                            T_max=EPOCH * len(train_dataloader))
    # # cosine LR 0.002 - 0.0004 with warmup
    # loss_func = nn.CrossEntropyLoss()
    loss_func = LabelSmoothing()
    loss_func = loss_func.to('cuda')

    Loss_list = []
    Accuracy_list = []
    train_Loss_list = []
    train_Accuracy_list = []
    # loss = []
    # loss1 = []
    if torch.cuda.is_available():
        model.cuda()

    if args.test_flag:
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('Epoch: {} '.format(start_epoch))
        # Val--------------------------------

        # val(model, val_dataloader, loss_func)
        if args.resume_net:
            print('Loading epoch {} ......'.format(start_epoch))

        else:
            return

    if os.path.exists(log_dir) and args.test_flag:
        checkpoint = torch.load(log_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('Loading epoch {} DONE！'.format(start_epoch))
    else:
        start_epoch = 1
        print('无保存模型，将从头开始训练！')


    """
    保存模型
    """
    torch.save(model.module.body.state_dict(),"./model/new_data.pth")


    train_acc_B = 0.
    num_B_Itr = len(train_dataloader)  
    num_Itr = num_B_Itr * EPOCH  # 总的 Itr
    N_Itr = 0  # 当前的Itr
    # Train--------------------------------
    for epoch in range(start_epoch, EPOCH + 1):
        since = time.time()
        print('epoch {}'.format(epoch))  # 显示每次训练次数
        # train_res(model, train_dataloader, epoch)

        model.train()
        # print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        optimizer.zero_grad()
        N_B_Itr = 0  # 每个Epoch当前的Itr

        for batch_x, batch_y in train_dataloader:
            since1 = time.time()

            batch_x = Variable(batch_x).cuda()
            batch_y = Variable(batch_y).cuda()

            # print("really")
            out = model(batch_x)
            # print('Go')
            # loss1 = loss_func(out, batch_y) ## CrossEn
            loss1 = loss_func(batch_x, out, batch_y)
            train_loss += loss1.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()

            loss1.backward()
            optimizer.step()
            lr_she.step()
            optimizer.zero_grad()

            # model_ema.update_parameters(model)

            N_Itr += 1
            N_B_Itr += 1
            Time_T = time.time() - since1  ## 当前Itr所花费的时间

            print(
                'Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Train Loss: {:.6f} || Acc: {:.6f} || Lr: {:.6f} || Time: {:.6f}s'.format(
                    epoch, EPOCH,
                    N_B_Itr, num_B_Itr,
                    N_Itr, num_Itr,
                    train_loss / (len(train_datasets)),
                    train_acc / (len(train_datasets)),
                    optimizer.param_groups[0]['lr'],
                    Time_T))  # 输出训练时的参数
        # print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_datasets)),
        #                                                train_acc / (len(train_datasets))))  # 输出训练时的loss和acc

        # save result
        import csv
        with open("Result_train.csv", "a", newline='') as f:
            writer = csv.writer(f)
            # writer.writerow(["Epoch    ", "     Train_loss", "     Train_acc"])
            # row = [epoch, ' ', train_loss / (len(train_datasets)), '    ', train_acc / (len(train_datasets))]
            row = ['Epoch: {}, Train Loss: {:.6f}, Acc: {:.6f}'.format(epoch, train_loss / (len(train_datasets)),
                                                                       train_acc / (len(train_datasets)))]
            writer.writerow(row)

        if train_acc_B < train_acc / (len(train_datasets)):
            # model_ema_state = get_model_state_dict(model_ema)
            # state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'EMA': model_ema_state}
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, weights + '/Mvitgcc_best.pth')
            train_acc_B = train_acc / (len(train_datasets))
            print('The best Model have been saved!')

        train_Loss_list.append(train_loss / (len(train_datasets)))
        train_Accuracy_list.append(100 * train_acc / (len(train_datasets)))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 输出训练和测试的时间

        # if (epoch % 20 == 0 and epoch > 0) or (epoch % 10 == 0 and epoch > decay1):
        if epoch > 400:
            print('评估模型')

            top1, loss11 = val(model, val_dataloader, loss_func)

            top1 = top1.__float__()
            loss11 = loss11.__float__()
            top1 = round(top1, 4)
            loss11 = round(loss11, 4)

            # save result
            with open("Result_val.csv", "a", newline='') as f:
                writer = csv.writer(f)
                # writer.writerow(["Epoch    ", "     Val_loss", "     Val_acc"])
                # row = [epoch, ' ', train_loss / (len(train_datasets)), '    ', train_acc / (len(train_datasets))]
                row = ['Epoch: {}, Val Loss: {:.6f}, Top1: {:.6f}'.format(epoch, top1, loss11)]
                writer.writerow(row)

            Loss_list.append(loss11)
            Accuracy_list.append(top1)

            print('保存模型')
            # model_ema_state = get_model_state_dict(model_ema)
            # state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'EMA': model_ema_state}
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, weights + '/Mvitgcc' + '_epoch_' + str(epoch) + '.pth')
        # if (epoch % 20 == 0 and epoch > 0) or (epoch % 10 == 0 and epoch > decay1):
        if (epoch % 20 == 0 and epoch > 0) or (epoch % 10 == 0 and epoch > decay1) or (epoch % 1 == 0 and epoch > decay2):

            print('保存模型')
            # model_ema_state = get_model_state_dict(model_ema)
            # state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'EMA': model_ema_state}
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, weights + '/Mvitgcc' + '_epoch_' + str(epoch) + '.pth')
    # y1 = Accuracy_list
    # y2 = Loss_list
    y3 = train_Accuracy_list
    y4 = train_Loss_list

    # x1 = range(len(Accuracy_list))
    # x2 = range(len(Loss_list))
    x3 = range(len(train_Accuracy_list))
    x4 = range(len(train_Loss_list))
    
    plt.subplot(2, 1, 1)
    plt.plot(x3, y3, '-')
    plt.title('Train accuracy vs. epoches')
    plt.ylabel('Train accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x4, y4, '-')
    plt.xlabel('Train loss vs. epoches')
    plt.ylabel('Train loss')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=350,help= 'Max number of epoch = 300')
    parser.add_argument('--decay1', type=int, default=150)
    parser.add_argument('--decay2',type=int,default=300)
    parser.add_argument('--batch_size', type=int, default=150)
    parser.add_argument('--lr', '--learning-rate', default=4e-4, type=float, help='max learning rate default = 8e-4')
    parser.add_argument('-init_lr',default=4e-6,type=float,help='init and min learning rate default = 8e-6')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for optimizer')
    parser.add_argument('--test_flag', type=bool, default=True, help='Val net or not')
    parser.add_argument('--resume_net', type=bool, default=True, help='Resume net for retraining')
    parser.add_argument('--network', default='mobilevit2_new',
                        help='Backbone network mobilevit2_1.00 or mobilevit2_0.75')
    parser.add_argument('--use_EMA', type=bool, default=False, help='Use EMA or not')

    parser.add_argument('--log_dir', type=str,
                        default="/media/amax/A23252133251ED33/QL_Model/CTD/weight_Cls/Mvitgcc_best.pth")
    parser.add_argument('--data_train_dir', type=str,
                        default="/media/amax/96EAAC13EAABEDA5/ImageNet1K/ILSVRC2012_img_train_P")
    parser.add_argument('--data_val_dir', type=str,
                        default="/media/amax/96EAAC13EAABEDA5/ImageNet1K/ILSVRC2012_img_val_P")

    parser.add_argument('--weights', type=str,
                        default='./weight_Cls', help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze_layers', type=bool, default=False)
    opt = parser.parse_args()

    main(opt)
