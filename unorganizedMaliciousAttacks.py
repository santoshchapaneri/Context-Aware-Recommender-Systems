#coding:utf-8
import random
import numpy as np
from attack import Attack


class UMAttack(Attack):
    def __init__(self,conf):
        super(UMAttack, self).__init__(conf)
        self.hotItems = sorted(self.itemProfile.iteritems(), key=lambda d: len(d[1]), reverse=True)[
                   :int(self.selectedSize * len(self.itemProfile))]

    def insertSpam(self,startID=0):
        print 'Modeling unorganized malicous attack...(include average attack, random attack and bandwagon attack)'
        itemList = self.itemProfile.keys()
        if startID == 0:
            self.startUserID = len(self.userProfile)
        else:
            self.startUserID = startID

        for i in range(int(len(self.userProfile)*self.attackSize)):
            #fill 
            
            selectedItems = self.getSelectedItems()
            fillerItems = self.getFillerItems()
            flag = random.uniform(0,1)
            if flag>0.6:
                for item in fillerItems:
                    self.spamProfile[str(self.startUserID)][str(itemList[item])] = round(self.itemAverage[str(itemList[item])])
            elif flag>0.4 and flag<=0.6:
                for item in fillerItems:
                    self.spamProfile[str(self.startUserID)][str(itemList[item])] = random.randint(self.minScore,self.maxScore)
            elif flag<=0.4:
                for item in fillerItems:
                    self.spamProfile[str(self.startUserID)][str(itemList[item])] = random.randint(self.minScore,self.maxScore)
                for item in selectedItems:
                    self.spamProfile[str(self.startUserID)][item] = self.targetScore
            #target 
            for j in range(self.targetCount):
                target = np.random.randint(len(self.targetItems))
                self.spamProfile[str(self.startUserID)][self.targetItems[target]] = self.targetScore
                self.spamItem[str(self.startUserID)].append(self.targetItems[target])
            self.startUserID += 1

    def getFillerItems(self):
        mu = int(self.fillerSize*len(self.itemProfile))
        sigma = int(0.1*mu)
        markedItemsCount = int(round(random.gauss(mu, sigma)))
        if markedItemsCount < 0:
            markedItemsCount = 0
        markedItems = np.random.randint(len(self.itemProfile), size=markedItemsCount)
        return markedItems

    def getSelectedItems(self):
        mu = int(self.selectedSize * len(self.itemProfile))
        sigma = int(0.1 * mu)
        markedItemsCount = abs(int(round(random.gauss(mu, sigma))))
        markedIndexes =  np.random.randint(len(self.hotItems), size=markedItemsCount)
        markedItems = [self.hotItems[index][0] for index in markedIndexes]
        return markedItems




