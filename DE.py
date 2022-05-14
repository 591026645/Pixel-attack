"""
差分进化方法
"""
import numpy as np
from IndividualEvaluation import EvaAttackEffect,EvaPerturbationEffect

DE_F = 0.33  # the scaling factor
DE_CR = 0.9  # the crossover probability

# 此函数中没有利用到个体的扰动强度，但依然将扰动强度的参数传入
def DE_Atk(NP, Gene, Population, ActFit, PerFit, TargetImage, currentGeneration):  # ActFit存储种群中所有个体的攻击效果，PerFit存储种群中所有个体的扰动强度
    MutantVector = np.zeros((NP, Gene))  # 存储产生的突变个体
    TrialVector = np.zeros((NP, Gene))  # 存储父代个体和突变个体交叉完之后，产生的子代个体
    Off_Ack = np.zeros(NP)  # 记录子代个体的攻击效果
    Off_Per = np.zeros(NP)  # 记录子代个体的扰动效果

    for i in range(NP):
        # mutation
        r0, r1, r2 = 0, 0, 0
        while r0 == i or r1 == i or r2 == i or r0 == r1 or r0 == r2 or r1 == r2:  # Ensure that i,r0,r1,r2 are different from each other
            r0 = np.random.randint(0, NP)  # base vector
            r1 = np.random.randint(0, NP)
            r2 = np.random.randint(0, NP)
        MutantVector[i] = Population[r0] + DE_F * (Population[r1] - Population[r2])

        # Crossover
        jRand = np.random.randint(0, Gene)
        for j in range(Gene):
            if np.random.random() < DE_CR or j == jRand:
                TrialVector[i, j] = MutantVector[i, j]
            else:
                TrialVector[i, j] = Population[i, j]

        # Map back to search space,这个只是单纯的考虑个体（噪音）的范围是否在合理的范围内
        for j in range(Gene):
            if TrialVector[i, j] + TargetImage[j] > 1:
                #TrialVector[i, j] = 1 - (TrialVector[i, j] - 1)
                TrialVector[i, j] = 1 - (TrialVector[i, j] + TargetImage[j] - 1) - TargetImage[j]
            elif TrialVector[i, j] < 0:
                TrialVector[i, j] = 0
            else:
                pass

        # 评价生成子代个体的攻击效果和攻击强度
        Off_Ack[i] = EvaAttackEffect(TrialVector[i], TargetImage)
        Off_Per[i] = EvaPerturbationEffect(TrialVector[i])

        # 对父代和子代进行判断，选择更优的个体存活到下一世代(此处没有考虑攻击强度下降的快慢)
        if (ActFit[i] >= Off_Ack[i]):
            Population[i] = TrialVector[i]
            ActFit[i] = Off_Ack[i]
            PerFit[i] = Off_Per[i]

    print("第" ,currentGeneration ,"世代")
    print("当前世代中最好攻击效果：{}".format(min(ActFit)))
    print("最好攻击效果的扰动程度：{}".format(PerFit[np.argmin(ActFit)]))
    #print("当前世代中最小扰动程度：{}".format(min(PerFit)))
    BEST_X = Population[np.argmin(ActFit)]
    currentGeneration += 1
    return BEST_X,Population


def DE_Re(NP, Gene, Population, ActFit, PerFit, TargetImage, currentGeneration):
    MutantVector = np.zeros((NP, Gene))  # 存储产生的突变个体
    TrialVector = np.zeros((NP, Gene))  # 存储父代个体和突变个体交叉完之后，产生的子代个体
    Off_Ack = np.zeros(NP)  # 记录子代个体的攻击效果
    Off_Per = np.zeros(NP)  # 记录子代个体的扰动效果

    for i in range(NP):
        # mutation
        r0, r1, r2 = 0, 0, 0
        while r0 == i or r1 == i or r2 == i or r0 == r1 or r0 == r2 or r1 == r2:  # Ensure that i,r0,r1,r2 are different from each other
            r0 = np.random.randint(0, NP)  # base vector
            r1 = np.random.randint(0, NP)
            r2 = np.random.randint(0, NP)
        MutantVector[i] = Population[r0] + DE_F * (Population[r1] - Population[r2])

        # Crossover
        jRand = np.random.randint(0, Gene)
        for j in range(Gene):
            if np.random.random() < DE_CR or j == jRand:
                TrialVector[i, j] = MutantVector[i, j]
            else:
                TrialVector[i, j] = Population[i, j]

        # Map back to search space,这个只是单纯的考虑个体（噪音）的范围是否在合理的范围内
        for j in range(Gene):
            if TrialVector[i, j] + TargetImage[j] > 1:
                #TrialVector[i, j] = 1 - (TrialVector[i, j] - 1)
                TrialVector[i, j] = 1 - (TrialVector[i, j] + TargetImage[j] - 1) - TargetImage[j]
            elif TrialVector[i, j] < 0:
                TrialVector[i, j] = 0
            else:
                pass

        # Set the disturbance value to 0 with a low probability
        for j in range(Gene):
            randNum = np.random.uniform(0, 1)
            if (randNum <= 0.2):
                TrialVector[i, j] = 0

        # 评价生成子代个体的恢复效果和扰动强度
        Off_Ack[i] = EvaAttackEffect(TrialVector[i], TargetImage)
        Off_Per[i] = EvaPerturbationEffect(TrialVector[i])

        # 对父代和子代进行判断，选择更优的个体存活到下一世代(此处没有考虑恢复强度下降的快慢)
        if (PerFit[i] >= Off_Per[i]):
            Population[i] = TrialVector[i]
            ActFit[i] = Off_Ack[i]
            PerFit[i] = Off_Per[i]

    print("第", currentGeneration, "世代")
    #print("当前世代中最好攻击效果：{}".format(min(ActFit)))
    print("最小扰动程度的攻击效果：{}".format(ActFit[np.argmin(PerFit)]))
    print("当前世代中最小扰动程度：{}".format(min(PerFit)))
    BEST_X = Population[np.argmin(PerFit)]
    currentGeneration += 1
    return BEST_X,Population
