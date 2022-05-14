"""
NSGA-Ⅱ
"""
import numpy as np
from IndividualEvaluation import EvaAttackEffect,EvaPerturbationEffect

DE_F = 0.33  # the scaling factor
DE_CR = 0.9  # the crossover probability

def function(Individual, TargetImage):
    y1 = EvaAttackEffect(Individual, TargetImage)
    y2 = EvaPerturbationEffect(Individual)
    return y1, y2

def fitness(Individual, NP, TargetImage):
    # 计算种群或者个体的适应度
    fits = np.array([function(Individual[i], TargetImage) for i in range(NP)])
    return fits

def nonDominationSort(pops, fits):
    """快速非支配排序算法
    Params:
        pops: 种群，nPop * nChr 数组
        fits: 适应度， nPop * nF 数组
    Return:
        ranks: 每个个体所对应的等级，一维数组
    """
    nPop = pops.shape[0]
    nF = fits.shape[1]  # 目标函数的个数
    ranks = np.zeros(nPop, dtype=np.int32)
    nPs = np.zeros(nPop)  # 每个个体p被支配解的个数
    sPs = []  # 每个个体支配的解的集合，把索引放进去
    for i in range(nPop):
        iSet = []  # 解i的支配解集
        for j in range(nPop):
            if i == j:
                continue
            isDom1 = fits[i] <= fits[j]
            isDom2 = fits[i] < fits[j]
            # 是否支配该解-> i支配j
            if sum(isDom1) == nF and sum(isDom2) >= 1:  #两个目标函数都支配且恢复目标完全支配
                if isDom2[0] == True and isDom2[1] == False:
                    nPs[i] += 1
                    print("lll")
                else:
                    iSet.append(j)
            # 是否被支配-> i被j支配
            if sum(~isDom2) == nF and sum(~isDom1) >= 1:
                if ~isDom1[0] == True and ~isDom1[1] == False:
                    iSet.append(j)
                    print("lll")
                else:
                    nPs[i] += 1
        sPs.append(iSet)  # 添加i支配的解的索引
    r = 0  # 当前等级为 0， 等级越低越好
    indices = np.arange(nPop)
    while sum(nPs==0) != 0:
        rIdices = indices[nPs==0]  # 当前被支配数为0的索引
        ranks[rIdices] = r
        for rIdx in rIdices:
            iSet = sPs[rIdx]
            nPs[iSet] -= 1
        nPs[rIdices] = -1  # 当前等级的被支配数设置为负数
        r += 1
    return ranks

def crowdingDistanceSort(pops, fits, ranks):
    """拥挤度排序算法
    Params:
        pops: 种群，nPop * nChr 数组
        fits: 适应度， nPop * nF 数组
        ranks：每个个体对应的等级，一维数组
    Return：
        dis: 每个个体的拥挤度，一维数组
    """
    nPop = pops.shape[0]
    nF = fits.shape[1]  # 目标个数
    dis = np.zeros(nPop)
    nR = ranks.max()  # 最大等级
    indices = np.arange(nPop)
    for r in range(nR+1):
        rIdices = indices[ranks==r]  # 当前等级种群的索引
        rPops = pops[ranks==r]  # 当前等级的种群
        rFits = fits[ranks==r]  # 当前等级种群的适应度
        rSortIdices = np.argsort(rFits, axis=0)  # 对纵向排序的索引
        rSortFits = np.sort(rFits,axis=0)
        fMax = np.max(rFits,axis=0)
        fMin = np.min(rFits,axis=0)
        n = len(rIdices)
        for i in range(nF):
            orIdices = rIdices[rSortIdices[:,i]]  # 当前操作元素的原始位置
            j = 1
            while n > 2 and j < n-1:
                if fMax[i] != fMin[i]:
                    dis[orIdices[j]] += (rSortFits[j+1,i] - rSortFits[j-1,i]) / \
                        (fMax[i] - fMin[i])
                else:
                    dis[orIdices[j]] = np.inf
                j += 1
            dis[orIdices[0]] = np.inf
            dis[orIdices[n-1]] = np.inf
    return dis

def optSelect(pops, fits, chrPops, chrFits, NP, Gene):
    """种群合并与优选
    Return:
        newPops, newFits
    """
    nF = fits.shape[1]
    newPops = np.zeros((NP, Gene))
    newFits = np.zeros((NP, nF))
    # 合并父代种群和子代种群构成一个新种群
    MergePops = np.concatenate((pops, chrPops), axis=0)
    MergeFits = np.concatenate((fits, chrFits), axis=0)
    MergeRanks = nonDominationSort(MergePops, MergeFits)
    MergeDistances = crowdingDistanceSort(MergePops, MergeFits, MergeRanks)

    indices = np.arange(MergePops.shape[0])
    r = 0
    i = 0
    rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
    while i + len(rIndices) <= NP:
        newPops[i:i + len(rIndices)] = MergePops[rIndices]
        newFits[i:i + len(rIndices)] = MergeFits[rIndices]
        r += 1  # 当前等级+1
        i += len(rIndices)
        rIndices = indices[MergeRanks == r]  # 当前等级为r的索引

    if i < NP:
        rDistances = MergeDistances[rIndices]  # 当前等级个体的拥挤度
        rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
        surIndices = rIndices[rSortedIdx[:(NP - i)]]
        newPops[i:] = MergePops[surIndices]
        newFits[i:] = MergeFits[surIndices]
    return (newPops, newFits)

def DE_Re(NP, Gene, Population, ActFit, PerFit, fits, TargetImage, currentGeneration):
    MutantVector = np.zeros((NP, Gene))  # 存储产生的突变个体
    TrialVector = np.zeros((NP, Gene))  # 存储父代个体和突变个体交叉完之后，产生的子代个体
    Off_Ack = np.zeros(NP)  # 记录子代个体的攻击效果
    Off_Per = np.zeros(NP)  # 记录子代个体的扰动效果
    ChPopulation = np.zeros((NP, Gene))  # 存储交叉后的子代

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
                TrialVector[i, j] = 1 - (TrialVector[i, j] + TargetImage[j] - 1) - TargetImage[j]
            elif TrialVector[i, j] < 0:
                TrialVector[i, j] = 0
            else:
                pass

        # 评价生成子代个体的恢复效果和扰动强度
        Off_Ack[i] = EvaAttackEffect(TrialVector[i], TargetImage)
        Off_Per[i] = EvaPerturbationEffect(TrialVector[i])

        ChPopulation[i, :] = TrialVector[i, :]

    # selection
    chrfits = fitness(ChPopulation, NP, TargetImage)
    Population, fits = optSelect(Population, fits, ChPopulation, chrfits, NP, Gene)  # 精英保留策略
    ranks = nonDominationSort(Population, fits)  # 非支配排序
    distances = crowdingDistanceSort(Population, fits, ranks)  # 拥挤度
    paretoFits = fits[ranks == 0]
    nf = paretoFits.shape[0]
    h1 = 0
    h2 = 0
    for i in range(nf):
        h1 += paretoFits[i, 0] / nf
        h2 += paretoFits[i, 1] / nf

    for i in range(NP):
        ActFit[i] = EvaAttackEffect(Population[i], TargetImage)
        PerFit[i] = EvaPerturbationEffect(Population[i])
    BEST_FIT = Population[np.argmin(ActFit)]
    BEST_DIS = Population[np.argmin(PerFit)]
    print("Recover: 第",currentGeneration,"世代中攻击值为：", EvaAttackEffect(BEST_DIS, TargetImage), "扰动最优值为：", EvaPerturbationEffect(BEST_DIS))
    #print("Recover: 第", currentGeneration, "世代 ", h1, " ", h2)
    return BEST_FIT, BEST_DIS


def nsga(NP, Gene, Population, ActFit, PerFit, TargetImage, currentGeneration):
    print("Recover")
    MAX_GENERATION = currentGeneration + 100
    fits = fitness(Population, NP, TargetImage)
    while True:
        BEST_FIT, BEST_DIS = DE_Re(NP, Gene, Population, ActFit, PerFit, fits, TargetImage, currentGeneration)
        currentGeneration += 1
        # if Dis(BEST_FIT) < D or currentGeneration > MAX_GENERATION:
        #if EvaAttackEffect(BEST_DIS, TargetImage) < F or currentGeneration > MAX_GENERATION:
        if currentGeneration > MAX_GENERATION:
            break

    ranks = nonDominationSort(Population, fits)  # 非支配排序
    distances = crowdingDistanceSort(Population, fits, ranks)  # 拥挤度
    paretoPops = Population[ranks == 0]
    paretoFits = fits[ranks == 0]
    # print(paretoPops, paretoFits)

    print("Best_fit ", "最优值：", EvaAttackEffect(BEST_FIT, TargetImage), "扰动程度：", EvaPerturbationEffect(BEST_FIT))
    BEST = BEST_FIT + TargetImage
    # result = model.predict(BEST.reshape(1, 784))
    # print("Fitness:",result[0])

    print("Best_dis ", "最优值：", EvaAttackEffect(BEST_DIS, TargetImage), "扰动程度：", EvaPerturbationEffect(BEST_DIS))
    BEST = BEST_DIS + TargetImage
    # result = model.predict(BEST.reshape(1, 784))
    # print("Fitness:",result[0])