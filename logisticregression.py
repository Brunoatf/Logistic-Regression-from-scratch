import numpy as np
from math import exp
from matplotlib import animation, pyplot as plt
from celluloid import Camera

class LogisticRegression:

    def sigmoid(self, inputs):
        return 1 / (1 + exp(-np.dot(inputs, self.coefficients) - self.bias))

    def normalize(self, inputs):
        return inputs - self.train_mean
    
    def predict(self, inputs):
        return 1 if self.sigmoid(inputs) >= 0.5 else 0

    def train(self, inputs, outputs, learning_rate, epochs, *generate_animation):

        '''Trains the model using a matrix of inputs - in which each line contains the data for one
        train case - and a vector of outputs. If the inputs have dimension 1 or 2, it is possible
        to set generate_animation true to generate a animation of the learning'''

        self.train_mean = np.mean(inputs)
        normalized_inputs = self.normalize(inputs)
        self.coefficients = np.zeros(inputs.shape[1]) #the values which multiply the input variables 
        self.bias = 0 #the bias is the linear coefficient of the linear equation used in the sigmoid's exponents

        if generate_animation:
            fig, ax = plt.subplots(1,2, figsize=[12,6])
            fig.suptitle("Logistic Regression training process:")
            ax[0].set_title("Mean squared error:")
            ax[1].set_title("Sigmoid curve:")
            camera = Camera(fig)
            mean_squared_errors = []

        for epoch in range(epochs): #for each epoch, we apply gradient descent for each training case:

            predicted_outputs = np.array([self.sigmoid(normalized_input) for normalized_input in normalized_inputs])

            if generate_animation:

                mean_squared_errors.append((sum(outputs-predicted_outputs)**2)/inputs.shape[0])
                ax[0].plot([i for i in range(1, len(mean_squared_errors)+1)], mean_squared_errors, color='black')
                ax[0].set_xlabel("Epoch")

                x = np.arange(normalized_inputs[0]-1, normalized_inputs[-1]+1, 0.1)
                ax[1].plot(x,np.array([self.sigmoid(value) for value in x]), color = 'black', label="Sigmoid curve")
                ax[1].scatter(normalized_inputs, outputs, c=outputs, cmap="coolwarm", label="Training points")
                ax[1].set_xlabel("Normalized input")
                ax[1].set_ylabel("Probability of belonging to the red class")
                camera.snap()

            self.bias -= learning_rate * -2 * sum((outputs - predicted_outputs) * predicted_outputs * (1 - predicted_outputs))
            dloss_dcoefficients = np.zeros(self.coefficients.size)
            for index in range(self.coefficients.size):
                dloss_dcoefficients[index] += -2 * sum((outputs - predicted_outputs) * predicted_outputs * (1 - predicted_outputs) * normalized_inputs[:,index])
            self.coefficients -= learning_rate * dloss_dcoefficients
        
        if generate_animation:
            anim = camera.animate()
            anim.save("animation_2d_logistic_regression.gif", fps=60)
        
    def test(self, inputs, expected_outputs):
        
        inputs = self.normalize(inputs)
        outputs = np.array([self.predict(input) for input in inputs])

        count = 0
        for i in range(inputs.shape[0]):
            if outputs[i] == expected_outputs[i]:
                count += 1
        
        accuracy = count / inputs.shape[0]
        print("Outputs:", outputs,"\nAccuracy:", accuracy)

model = LogisticRegression()

model.train(np.array([[1],[2],[3],[4],[5],[6],[7]]), np.array([1,1,1,0,0,0,0]), 0.1, 300, True)
model.test(np.array([[1],[2],[3],[4],[5],[6],[7],[8]]), np.array([0,0,0,1,1,1,1,1]))

#Nota: se os dados de treino estiverem perfeitamente balanceados, a derivada do loss será 0 e vai dar merda