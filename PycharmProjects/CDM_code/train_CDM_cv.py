import argparse
import os
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as Data
from pytorch_transformers import *
from torch.autograd import Variable
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, multilabel_confusion_matrix

from load_data_CDM import *
from mixtext import MixText


parser = argparse.ArgumentParser(description='PyTorch MixText')

parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--lrmain', '--learning-rate-bert', default=1e-5, type=float, #[1e-5,5e-5]
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=1e-5
                    , type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--n-labeled', type=int, default=20,
                    help='number of labeled data')

parser.add_argument('--prefix', type= bool, default= False,
                    help='number of labeled data')

parser.add_argument('--val-iteration', type=int, default=20,
                    help='number of labeled data')


parser.add_argument('--mix-option', default=True, type=bool, metavar='N',
                    help='mix option, whether to mix or not')
parser.add_argument('--mix-method', default=0, type=int, metavar='N',
                    help='mix method, set different mix method')

parser.add_argument('--train_aug', default=False, type=bool, metavar='N',
                    help='augment labeled training data')


parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='pretrained model')

parser.add_argument('--data-path', type=str, default='yahoo_answers_csv/',
                    help='path to data folders')

parser.add_argument('--mix-layers-set', nargs='+',
                    default=[0, 1, 2, 3], type=int, help='define mix layer set')

parser.add_argument('--alpha', default=0.75, type=float,
                    help='alpha for beta distribution')

parser.add_argument('--T', default=0.5, type=float,
                    help='temperature for sharpen function')


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

best_acc = 0
best_micro = 0
best_macro = 0
total_steps = 0
flag = 0
print('Whether mix: ', args.mix_option)
print("Mix layers sets: ", args.mix_layers_set)

def main():
    global best_acc
    global best_micro
    global best_macro
    # Read dataset and build dataloaders
    data_path = './data/'
    train_df = pd.read_csv(data_path+'full_data.csv', header = 0)
    train_df.drop('Unnamed: 0', inplace=True, axis=1)
    #
    # test_df = pd.read_csv(data_path+'test.csv', header = 0)
    # test_df.drop('Unnamed: 0', inplace=True, axis=1)


    # train_labeled_set, test_set, n_labels = get_data(args.n_labeled,max_seq_len=256,model=args.model, train_aug=args.train_aug)
    # # labeled_trainloader = Data.DataLoader(
    # #     dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    # # unlabeled_trainloader = Data.DataLoader(
    # #     dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True)
    # # val_loader = Data.DataLoader(
    # #     dataset=val_set, batch_size=2, shuffle=False)
    # test_loader = Data.DataLoader(
    #     dataset=test_set, batch_size=2, shuffle=False)

    # Define the model, set the optimizer
    model = MixText(10, args.mix_option).cuda()
    model = nn.DataParallel(model)


    num_warmup_steps = math.floor(50)
    num_total_steps = args.val_iteration



    # train_criterion = SemiLoss()
    criterion =nn.BCEWithLogitsLoss()


    val_accs = []
    val_micros = []
    val_macros = []
    kf5 = KFold(n_splits=5,shuffle=False) #5-fold cross validation

    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": args.lrmain},
            {"params": model.module.linear.parameters(), "lr": args.lrlast},
        ])
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    # Start training
    for train_index, val_index in kf5.split(train_df):


        X_train = train_df.iloc[train_index]
        X_val = train_df.iloc[val_index]
        X_train, X_val = X_train.reset_index(drop=True), X_val.reset_index(drop=True)
        train_labeled_dataset,val_dataset,n_labels = get_data(X_train,X_val,max_seq_len=256,
                                                                    model=args.model,train_aug = False, prefix = args.prefix)

        labeled_trainloader = Data.DataLoader(
            dataset=train_labeled_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = Data.DataLoader(
            dataset=val_dataset, batch_size=3, shuffle=False)
        # test_loader = Data.DataLoader(
        #     dataset=test_set, batch_size=2, shuffle=False)
        for epoch in range(args.epochs):
            model = train(labeled_trainloader, model, optimizer,
              scheduler, criterion, epoch, n_labels, args.train_aug)

        scheduler.step()

        # _, train_acc = validate(labeled_trainloader,
        #                        model,  criterion, epoch, mode='Train Stats')
        #print("epoch {}, train acc {}".format(epoch, train_acc))

        val_acc, val_micro,val_macro = validate(
            val_loader, model, criterion, epoch, mode='Valid Stats')

        print("epoch {}, val acc {}, val_micro {},val_macro {}".format(
            epoch, val_acc, val_micro,val_macro))
        val_accs.append(val_acc)
        val_micros.append(val_micro)
        val_macros.append(val_macro)
        # if val_micro >= best_micro:
        #     best_micro = val_micro
        #     test_acc, test_micro, test_macro = validate(
        #             test_loader, model, criterion, epoch, mode='Test Stats ')
        #     test_micros.append(test_micro)
        #
        # if val_macro >= best_macro:
        #     best_macro = val_macro
        #     test_acc, test_micro, test_macro = validate(
        #             test_loader, model, criterion, epoch, mode='Test Stats ')
        #     test_micros.append(test_micro)
        # if val_acc >= best_acc:
        #     best_acc = val_acc
        #     test_acc, test_micro,test_macro = validate(
        #         test_loader, model, criterion, epoch, mode='Test Stats ')
        #     test_accs.append(test_acc)
        #     test_micros.append(test_micro)
        #     test_macros.append(test_macro)
        #     # test_micro.append(test_micro)
        #     # test_macro.append(test_macro)

        model.apply(weight_reset)

        print('Epoch: ', epoch)

        # print('Best acc:')
        # print(best_acc)


    print("Finished training!")
    # print('Best acc:')
    # print(best_acc)

    print("Average val F1_micro",np.mean(val_micros))
    print("Average val F1_macro", np.mean(val_macros))
    print("Average val acc", np.mean(val_accs))


#
def weight_reset(m):
  for layer in m.children():
    if hasattr(layer, 'reset_parameters'):
      layer.reset_parameters()

def train(labeled_trainloader,model, optimizer, scheduler, criterion, epoch, n_labels, train_aug=False):
    labeled_train_iter = iter(labeled_trainloader)
    model.train()

    global total_steps

    for batch_idx in range(args.val_iteration):

        total_steps += 1


        if not train_aug:
            try:
                inputs_x, targets_x, inputs_x_length = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x, inputs_x_length = labeled_train_iter.next()
        else:
            try:
                (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length,
                                                           inputs_x_length_aug) = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length,
                                                           inputs_x_length_aug) = labeled_train_iter.next()

        batch_size = inputs_x.size(0)
        batch_size_2 = inputs_x.size(0)

        # targets_x = torch.zeros(batch_size, n_labels).scatter_(
        #     1, targets_x)

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        #

        # print('input',inputs_x.shape)
        # print('target:',targets_x.shape)
        # inputs_u = inputs_u.cuda()
        # inputs_u2 = inputs_u2.cuda()
        # inputs_ori = inputs_ori.cuda()

        mask = []

        # with torch.no_grad():
        #     # Predict labels for unlabeled data.
        #     outputs_u = model(inputs_u)
        #     outputs_u2 = model(inputs_u2)
        #     outputs_ori = model(inputs_ori)
            #
            # # Based on translation qualities, choose different weights here.
            # # For AG News: German: 1, Russian: 0, ori: 1
            # # For DBPedia: German: 1, Russian: 1, ori: 1
            # # For IMDB: German: 0, Russian: 0, ori: 1
            # # For Yahoo Answers: German: 1, Russian: 0, ori: 1 / German: 0, Russian: 0, ori: 1
            # p = (0 * torch.softmax(outputs_u, dim=1) + 0 * torch.softmax(outputs_u2,
            #                                                              dim=1) + 1 * torch.softmax(outputs_ori, dim=1)) / (1)
            # # Do a sharpen here.
            # pt = p**(1/args.T)
            # targets_u = pt / pt.sum(dim=1, keepdim=True)
            # targets_u = targets_u.detach()

        mixed = 1

        if args.co:
            mix_ = np.random.choice([0, 1], 1)[0]
        else:
            mix_ = 1

        if mix_ == 1:
            l = np.random.beta(args.alpha, args.alpha)
            if args.separate_mix:
                l = l
            else:
                l = max(l, 1-l)
        else:
            l = 1

        mix_layer = np.random.choice(args.mix_layers_set, 1)[0]
        mix_layer = mix_layer - 1

        if not train_aug:
            # print('yes')
            all_inputs = inputs_x
            all_lengths = inputs_x_length
            all_targets = targets_x
            # all_inputs = torch.cat(
            #     [inputs_x, inputs_u, inputs_u2, inputs_ori, inputs_ori], dim=0)
            #
            # all_lengths = torch.cat(
            #     [inputs_x_length, length_u, length_u2, length_ori, length_ori], dim=0)
            #
            # all_targets = torch.cat(
            #     [targets_x, targets_u, targets_u, targets_u, targets_u], dim=0)

        else:
            all_inputs = torch.cat([inputs_x,inputs_x_aug],dim=0)
            all_lengths = torch.cat([inputs_x_length,inputs_x_length_aug],dim=0)
            all_targets = torch.cat([targets_x,targets_x],dim=0)
            # all_inputs = torch.cat(
            #     [inputs_x, inputs_x_aug, inputs_u, inputs_u2, inputs_ori], dim=0)
            # all_lengths = torch.cat(
            #     [inputs_x_length, inputs_x_length, length_u, length_u2, length_ori], dim=0)
            # all_targets = torch.cat(
            #     [targets_x, targets_x, targets_u, targets_u, targets_u], dim=0)

        if args.separate_mix:
            idx1 = torch.randperm(batch_size)
            idx2 = torch.randperm(all_inputs.size(0) - batch_size) + batch_size
            idx = torch.cat([idx1, idx2], dim=0)

        else:
            idx1 = torch.randperm(all_inputs.size(0) - batch_size_2)
            idx2 = torch.arange(batch_size_2) + \
                all_inputs.size(0) - batch_size_2
            idx = torch.cat([idx1, idx2], dim=0)


        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        length_a, length_b = all_lengths, all_lengths[idx]

        if args.mix_method == 0:
            # Mix sentences' hidden representations
            # print(0)
            logits = model(input_a, input_b, l, mix_layer)
            # print('logit',logits[:batch_size].shape)
            mixed_target = l * target_a + (1 - l) * target_b
            # print(mixed_target.shape)

        # elif args.mix_method == 1:
        #     # Concat snippet of two training sentences, the snippets are selected based on l
        #     # For example: "I lova you so much" and "He likes NLP" could be mixed as "He likes NLP so much".
        #     # The corresponding labels are mixed with coefficient as well
        #     mixed_input = []
        #     if l != 1:
        #         for i in range(input_a.size(0)):
        #             length1 = math.floor(int(length_a[i]) * l)
        #             idx1 = torch.randperm(int(length_a[i]) - length1 + 1)[0]
        #             length2 = math.ceil(int(length_b[i]) * (1-l))
        #             if length1 + length2 > 256:
        #                 length2 = 256-length1 - 1
        #             idx2 = torch.randperm(int(length_b[i]) - length2 + 1)[0]
        #             try:
        #                 mixed_input.append(
        #                     torch.cat((input_a[i][idx1: idx1 + length1], torch.tensor([102]).cuda(), input_b[i][idx2:idx2 + length2], torch.tensor([0]*(256-1-length1-length2)).cuda()), dim=0).unsqueeze(0))
        #             except:
        #                 print(256 - 1 - length1 - length2,
        #                       idx2, length2, idx1, length1)
        #
        #         mixed_input = torch.cat(mixed_input, dim=0)
        #
        #     else:
        #         mixed_input = input_a
        #
        #     logits = model(mixed_input)
        #     mixed_target = l * target_a + (1 - l) * target_b
        #
        # elif args.mix_method == 2:
        #     # Concat two training sentences
        #     # The corresponding labels are averaged
        #     if l == 1:
        #         mixed_input = []
        #         for i in range(input_a.size(0)):
        #             mixed_input.append(
        #                 torch.cat((input_a[i][:length_a[i]], torch.tensor([102]).cuda(), input_b[i][:length_b[i]], torch.tensor([0]*(512-1-int(length_a[i])-int(length_b[i]))).cuda()), dim=0).unsqueeze(0))
        #
        #         mixed_input = torch.cat(mixed_input, dim=0)
        #         logits = model(mixed_input, sent_size=512)
        #
        #         #mixed_target = torch.clamp(target_a + target_b, max = 1)
        #         mixed = 0
        #         mixed_target = (target_a + target_b)/2
        #     else:
        #         mixed_input = input_a
        #         mixed_target = target_a
        #         logits = model(mixed_input, sent_size=256)
        #         mixed = 1
        loss = criterion(logits, mixed_target)



        # max_grad_norm = 1.0
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        print("epoch {} loss {}".format(
                epoch,loss.item()))
        return model


def validate(valloader, model, criterion, epoch, mode):
    model.eval()
    with torch.no_grad():
        # loss_total = 0
        # total_sample = 0
        # acc_total = 0
        # correct = 0

        for batch_idx, (inputs, targets, length) in enumerate(valloader):
            # print(inputs)
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            targets = targets.cpu().detach().numpy().tolist()
            targets = np.nan_to_num(targets)
            outputs = model(inputs)
            outputs = outputs.detach().cpu().clone().numpy()
            # print(outputs.type)
            # print('outputs_ori',outputs)
            outputs = (outputs >= 0)
            # print('outputs:',outputs)
            # print('targets:',targets)
            accuracy = metrics.accuracy_score(targets[0], outputs[0])
            f1_score_micro = metrics.f1_score(targets[0], outputs[0], average='micro',labels=np.unique(targets))
            f1_score_macro = metrics.f1_score(targets[0], outputs[0], average='macro',labels=np.unique(targets))

        #     _, predicted = torch.max(outputs.data, 1)
        #
        #     if batch_idx == 0:
        #         print("Sample some true labeles and predicted labels")
        #         print('predicted:',predicted[:20])
        #         print('target:',targets[:20])
        #
        #     correct += (np.array(predicted.cpu()) ==
        #                 np.array(targets.cpu())).sum()
        #     loss_total += loss.item() * inputs.shape[0]
        #     total_sample += inputs.shape[0]
        #
        # acc_total = correct/total_sample
        # loss_total = loss_total/total_samp    print('Test micro:')
    # print(test_micro)
    #
    # print('Test macro:')
    # print(test_macro)

    return accuracy, f1_score_micro,f1_score_macro


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, targets_u, mixed=1):

        if args.mix_method == 0 or args.mix_method == 1:

            Lx = - \
                torch.mean(torch.sum(F.log_softmax(
                    outputs_x, dim=1) * targets_x, dim=1))

            probs_u = torch.softmax(outputs_u, dim=1)

            Lu = F.kl_div(probs_u.log(), targets_u, None, None, 'batchmean')

            Lu2 = torch.mean(torch.clamp(torch.sum(-F.softmax(outputs_u, dim=1)
                                                   * F.log_softmax(outputs_u, dim=1), dim=1) - args.margin, min=0))

        elif args.mix_method == 2:
            if mixed == 0:
                Lx = - \
                    torch.mean(torch.sum(F.logsigmoid(
                        outputs_x) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)

                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
                    F.softmax(outputs_u_2, dim=1) * F.softmax(outputs_u_2, dim=1), dim=1), min=0))
            else:
                Lx = - \
                    torch.mean(torch.sum(F.log_softmax(
                        outputs_x, dim=1) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)
                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
                    F.softmax(outputs_u, dim=1) * F.softmax(outputs_u, dim=1), dim=1), min=0))

        return Lx, Lu, args.lambda_u * linear_rampup(epoch), Lu2, args.lambda_u_hinge * linear_rampup(epoch)


if __name__ == '__main__':
    main()
