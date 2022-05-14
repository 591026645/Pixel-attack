"""
计算攻击强度与扰动程度
"""
from keras.models import load_model
model = load_model('mnist_model.h5')

# Return the predicted value of the model for the adversarial sample (= the original sample + a noise).
def EvaAttackEffect(Individual, Original):
    Adversarial = Individual[:] + Original[:]
    result = model.predict(Adversarial.reshape(1, 784))
    return result[0][7]

# Calculate the intensity of the noise
def EvaPerturbationEffect(Individual):
    result = 0
    for i in range(784):
        l = Individual[i]
        result += l ** 2
    return result