import Utils.TimeLogger as logger
import torch as t
from Utils.TimeLogger import log
from Params import args
from Model import Model, MLPNet
from DataHandler import DataHandler
import numpy as np
import pickle
from Utils.Utils import *
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
import setproctitle
import pretrainTeacher as pre

class Coach:
    def __init__(self, handler, model_name=None):
        self.handler = handler
        self.model_name = model_name

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if self.model_name:
            self.loadModel(self.model_name)
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:
                reses = self.testEpoch()
                log(self.makePrint('Test', ep, reses, tstFlag))
                self.saveHistory()
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        self.saveHistory()

    def prepareModel(self):
        teacher = self.loadTeacher()
        self.model = Model(teacher).cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

    def trainEpoch(self):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch
        for i, tem in enumerate(trnLoader):
            ancs, poss, negs = tem
            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()
            loss, losses = self.model.calcLoss(self.handler.torchBiAdj, ancs, poss, negs, self.opt)
            epLoss += loss.item()
            epPreLoss += losses['mainLoss'].item()
            # regLoss = losses['regLoss'].item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            log('Step %d/%d: loss = %.3f        ' % (i, steps, loss), save=False, oneline=True)
            # log('Step %d/%d: loss = %.3f, regLoss = %.3f         ' % (i, steps, loss, regLoss), save=False, oneline=True)
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epRecall, epNdcg = [0] * 2
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()

            allPreds = self.model.student.testPred(usr, trnMask)
            _, topLocs = t.topk(allPreds, args.topk)
            recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
            epRecall += recall
            epNdcg += ndcg
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
        ret = dict()
        ret['Recall'] = epRecall / num
        ret['NDCG'] = epNdcg / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def loadTeacher(self):
        ckp = t.load('/home/chy/SimRec-main/Models/teacher_' + args.teacher_model + '.mod')
        teacher = ckp['model']
        return teacher

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('/home/chy/SimRec-main/History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        content = {
            'model': self.model,
        }
        t.save(content, '/home/chy/SimRec-main/Models/' + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def loadModel(self, model_name):
        ckp = t.load('/home/chy/SimRec-main/Models/' + model_name + '.mod')
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        with open('/home/chy/SimRec-main/History/' + model_name + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.saveDefault = True

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')
    
    recall_results = {}
    ndcg_results = {}

    t_recall={}
    t_ndcg={}
    Tea=True
    #绘制曲线图
    if args.load_model:
        for model_name in args.load_model:
            log(f'Processing model: {model_name}')
            coach = Coach(handler, model_name)
            coach.run()
            recall_results[model_name] = coach.metrics['TestRecall']
            ndcg_results[model_name] = coach.metrics['TestNDCG']
            if Tea:
                coach1 = pre.Coach(handler, model_name)
                coach1.run()
                model_name='teacher_'+ model_name
                print(model_name)
                log(f'Processing model: {model_name}')
                t_recall[model_name] = coach1.metrics['TestRecall']
                t_ndcg[model_name] = coach1.metrics['TestNDCG']
                Tea=False

    else:
        coach = Coach(handler)
        coach.run()
        recall_results['default'] = coach.metrics['TestRecall']
        ndcg_results['default'] = coach.metrics['TestNDCG']

    # 绘制对比图
    epochs = range(len(next(iter(recall_results.values()))))  # 使用第一个模型的长度作为参考


    plt.figure(figsize=(12, 5))

    # Plot Recall
    plt.subplot(1, 2, 1)
    first = True  # 标志变量，用于区分第一个模型
    for model_name, recalls in recall_results.items():
        if first:
            plt.plot(epochs, recalls, label=model_name, linestyle='-')
            for model_name, recall1 in t_recall.items():
                plt.plot(epochs, recall1, label=model_name, linestyle='-')
            first = False
        else:
            plt.plot(epochs, recalls, label=model_name, linestyle='--')  # 其他模型绘制虚线
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('amazon')
    plt.legend()

    # Plot NDCG
    plt.subplot(1, 2, 2)
    first = True  # 重新设置标志变量，用于NDCG的图
    for model_name, ndcgs in ndcg_results.items():
        if first:
            plt.plot(epochs, ndcgs, label=model_name, linestyle='-')  # 第一个模型绘制实线
            for model_name, ndcg1 in t_ndcg.items():
                plt.plot(epochs, ndcg1, label=model_name, linestyle='-')
            first = False
        else:
            plt.plot(epochs, ndcgs, label=model_name, linestyle='--')  # 其他模型绘制虚线
    plt.xlabel('Epoch')
    plt.ylabel('NDCG')
    plt.title('amazon')
    plt.legend()

    plt.tight_layout()
    save_path = "./img/recall_ndcg_comparison.png"
    plt.savefig(save_path)
    plt.show()

