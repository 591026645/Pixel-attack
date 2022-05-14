import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import load_model
from IndividualEvaluation import EvaAttackEffect,EvaPerturbationEffect
from DE import DE_Atk,DE_Re

NP = 100  # population size
Gene = 784   # Image size
xMin = 0   # the lower limit of the search space (Normalized)
xMax = 255 # the upper limit of the search space (Normalized)
DE_F = 0.33  #  the scaling factor
DE_CR = 0.1  # the crossover probability
MAX_GENERATION = 1000  # the maximum number of iterations
TRIAL_NUM = 1 # the number of experiments to be repeated
currentGeneration = 0
BestStore = []  # store the BestFound
BestFound = 0  # Record the images with the best attack effect and the lowest disturbance degree found so far
BEST_X = 0
Db = np.zeros(Gene)# 记录上一代的BEST_X
Fb = np.zeros(Gene) # 记录上一代的BestFound

Population = np.zeros((NP, Gene)) # population
Obj_Ack = np.zeros(NP) # Record the attack effects of individuals
Obj_Per = np.zeros(NP) # Record the perturbation intensity of individuals

# 数据处理
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)  # Flatten the picture into a vector
x_test = x_test.reshape(10000, 784)  # Do the same for the test-set
x_train = x_train / xMax
x_test = x_test / xMax
model = load_model('mnist_model.h5')

def CreateNN():
    pass

# find best individual
def BestIndividual():
    global BEST_X, BestFound
    for i in range(NP):
        Obj_Ack[i] = calAttackEffect(Population[i], x_test[0])
    # Find the best seed with the lowest fitness degree
    BEST_X = Population[np.argmin(Obj_Ack)]
    BestFound = Population[np.argmin(Obj_Ack)]

# Calculate attack effect and perturbation intensity of the incoming individual
def EvaluateIndividual(Fb, Db):
    #global Obj_Ack, Obj_Per, BestFound, currentGeneration
    s_ = 1
    BestFound[:] = Fb[:]
    BEST_X = Db[:]

    if calAttackEffect(BestFound, x_test[0]) > 0.4:
        s_ -= 1
        BestFound[:] = BEST_X[:]

    elif calAttackEffect(BEST_X, x_test[0]) <= 0.4:
        for i in range(NP):
            if Obj_Ack[i] < calAttackEffect(BestFound, x_test[0]) and Obj_Per[i] < Dis(BestFound):  #输出最小扰动程度
            #if FITS[i] < calAttackEffect(BestFound, x_test[0]) and NUM[i] < num(BestFound):  # 输出最小扰动个数
                BestFound[:] = Population[i, :]
                s_ -= 1
                # print(calAttackEffect(BestFound, x_test[0]), Dis(BestFound))
    if s_ == 1:
        BestFound[:] = Fb[:]

    print("   BestFound ", "最优值：", calAttackEffect(BestFound, x_test[0]), "扰动程度：", Dis(BestFound))
    # BestStore.append((currentGeneration, calAttackEffect(BestFound, x_test[0]), Dis(BestFound)))
    with open("AS1.txt", "a", encoding='utf-8') as f:
        f.write("最优值：" + str(calAttackEffect(BestFound, x_test[0])) + "   " + "扰动程度：" + str(
            Dis(BestFound)) + "   " + "代数：" + str(currentGeneration) + "\r\n")

# Pop_Num: population size, Gene_size: GENE size, Pop: populations
def Poplulation_Init(Pop_Num, Gene_size, Pop):
    for r in range(0, Pop_Num):
        for i in range(0, Gene_size):
            randNum = np.random.uniform(0, 1)
            if randNum <= 0.1: # Generate noise with 5% probability
                noise = 0.1*np.random.normal(0, 1)  # Add noise and randomly generate attack samples
                if noise < 0.0:
                    Pop[r, i] = 0.0
                elif noise > 1.0:
                    Pop[r, i] = 1.0
                else:
                    Pop[r, i] = noise
            else:
                Pop[r, i] =0.0

def Attack_Stage():
    DE_Atk(NP, Gene, DE_F, DE_CR, x_test, Obj_Ack, Obj_Per, BEST_X, Population, BestFound, currentGeneration,Fb, Db)
    BestStore.append((currentGeneration-1, calAttackEffect(BEST_X, x_test[0]), Dis(BEST_X)))
    EvaluateIndividual(Fb, Db)

def Recovery_Stage():
    DE_Re(NP, Gene, DE_F, DE_CR, x_test, Obj_Ack, Obj_Per, BEST_X, Population, BestFound, currentGeneration,Fb, Db)
    BestStore.append((currentGeneration-1, calAttackEffect(BEST_X, x_test[0]), Dis(BEST_X)))
    EvaluateIndividual(Fb, Db)

def main():
    global NP,Gene,Population, currentGeneration, Obj_Ack, Obj_Per,MAX_GENERATION, TRIAL_NUM
    for TrialNum in range(0, TRIAL_NUM):
        np.random.seed(TrialNum)  # fix the seed for generating random numbers to reproduce experiments
        Poplulation_Init(NP, Gene, Population) # randomly initialize the population
        BestIndividual()
        flag=True
        Age_Ack = np.mean(Obj_Ack)
        Age_Per = np.mean(Obj_Per)
        while currentGeneration < MAX_GENERATION: # the termination condition is satisfied
            currentGeneration += 1
            if flag: #
                Attack_Stage()
                Cur_Ack = np.mean(Obj_Ack)
                if(Cur_Ack < Age_Ack*0.9):
                    flag = False
                    Age_Per = np.mean(Obj_Per)
                    Age_Ack = Age_Ack * 0.9
                    print("Re")
            else:
                Recovery_Stage()
                Cur_Per = np.mean(Obj_Per)
                if(Cur_Per < Age_Per*0.8):
                    flag = True
                    #Age_Ack = np.mean(Obj_Ack)
                    #Age_Per = Age_Per * 0.8
                    print("Atk")

        print("Best ", "最优值：", calAttackEffect(BestFound, x_test[0]), "扰动程度：", Dis(BestFound))
        BEST = BestFound + x_test[0]
        result = model.predict(BEST.reshape(1, 784))
        print("Fitness:", result[0])
        plt.imshow(BEST.reshape(28, 28), cmap='gray')
        plt.show()

        x_data = []
        y_data1 = []
        y_data2 = []
        for i in range(MAX_GENERATION):
            x_data.append(BestStore[i][0])  # currentGeneration
            y_data1.append(BestStore[i][1])  # fitness degree
            y_data2.append(BestStore[i][2])  # disturbance degree
        plt.plot(x_data, y_data1, color='red', linewidth=2.0, linestyle='-')
        plt.show()
        plt.plot(x_data, y_data2, color='blue', linewidth=2.0, linestyle='-')
        plt.show()

if __name__ == "__main__":
    main()