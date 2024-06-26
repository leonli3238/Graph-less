import torch as t
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import LightGCN
from DataHandler import DataHandler
import numpy as np
import pickle
from Utils.Utils import *
import os
import setproctitle
import matplotlib.pyplot as plt

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
            print()
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        self.saveHistory()

    def prepareModel(self):
        self.model = LightGCN().cuda()
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

            uEmbeds, iEmbeds = self.model(self.handler.torchBiAdj)
            mainLoss = -self.model.pairPredictwEmbeds(uEmbeds, iEmbeds, ancs, poss, negs).sigmoid().log().mean()
            regLoss = calcRegLoss(model=self.model) * args.reg
            loss = mainLoss + regLoss

            epLoss += loss.item()
            epPreLoss += mainLoss.item()
            regLoss = regLoss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            log('Step %d/%d: loss = %.3f, regLoss = %.3f         ' % (i, steps, loss, regLoss), save=False, oneline=True)
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

            allPreds = self.model.testPred(usr, trnMask, self.handler.torchBiAdj)
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
        recallBig = 0
        ndcgBig =0
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

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('/home/chy/SimRec-main/History/teacher_' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        content = {
            'model': self.model,
        }
        t.save(content, '/home/chy/SimRec-main/Models/teacher_' + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def loadModel(self, model_name):
        ckp = t.load('/home/chy/SimRec-main/Models/teacher_' +  model_name  + '.mod')
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        with open('/home/chy/SimRec-main/History/teacher_' + model_name + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')	

    def plotResults(self):
        epochs = range(len(self.metrics['TestRecall']))
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)

        plt.plot(epochs, self.metrics['TestRecall'], label='Test Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.title('Recall over Epochs')
        plt.legend()

        plt.subplot(1, 2, 2)

        plt.plot(epochs, self.metrics['TestNDCG'], label='Test NDCG')
        plt.xlabel('Epoch')
        plt.ylabel('NDCG')
        plt.title('NDCG over Epochs')
        plt.legend()
        
        plt.tight_layout()
        save_path = "./img/recall_ndcg2.png"
        plt.savefig(save_path)
        plt.show()
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.saveDefault = True
    
    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')
    recall_results = {}
    ndcg_results = {}
    
    if args.load_model:
        for model_name in args.load_model:
            log(f'Processing model: {model_name}')
            coach = Coach(handler, model_name)
            coach.run()
            recall_results[model_name] = coach.metrics['TestRecall']
            ndcg_results[model_name] = coach.metrics['TestNDCG']
    else:
        coach = Coach(handler)
        coach.run()
        recall_results['default'] = coach.metrics['TestRecall']
        ndcg_results['default'] = coach.metrics['TestNDCG']