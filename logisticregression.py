import numpy as np
from math import exp
from matplotlib import animation, pyplot as plt
from celluloid import Camera

class LogisticRegression:

    def sigmoid(self, inputs):
        return 1 / (1 + np.exp(-np.dot(inputs, self.coefficients) - self.bias))

    def normalize(self, inputs):
        return inputs - self.train_mean
    
    def predict(self, inputs):
        return 1 if self.sigmoid(inputs) >= 0.5 else 0

    def train(self, inputs, outputs, learning_rate, epochs, *display):

        '''Trains the model using a matrix of inputs - in which each line contains the data for one
        train case - and a vector of outputs. If the inputs have dimension 1 or 2, it is possible
        to set display true to generate a animation of the learning process.'''

        self.train_mean = np.mean(inputs)
        normalized_inputs = self.normalize(inputs)
        self.coefficients = np.zeros(inputs.shape[1]) #the values which multiply the input variables 
        self.bias = 0 #the bias is the linear coefficient of the linear equation used in the sigmoid's exponents

        if display:
            fig = plt.figure(figsize=[12,6])
            fig.suptitle("Logistic Regression training process:")
            mse_graph = fig.add_subplot(1,2,1)
            mse_graph.set_title("Mean squared error:")
            mse_graph.set_xlabel("Epoch")
            if inputs.shape[1] == 1:
                model_graph = fig.add_subplot(1,2,2)
            else:
                model_graph = fig.add_subplot(1,2,2, projection='3d')
                aux = np.linspace(-10, 10, 20)
                x, y = np.meshgrid(aux, aux)
                model_graph.set_xlabel("Normalized x")
                model_graph.set_xticks(np.linspace(-10,10,5))
                model_graph.set_ylabel("Normalized y")
                model_graph.set_yticks(np.linspace(-10,10,5))
                model_graph.set_zlabel("Probability of belonging to the red class")
            model_graph.set_title("Sigmoid curve:")
            camera = Camera(fig)
            mean_squared_errors = np.array([])

        for epoch in range(epochs): #for each epoch, we apply gradient descent for each training case:

            predicted_outputs = np.array([self.sigmoid(normalized_input) for normalized_input in normalized_inputs])

            if display:

                mean_squared_errors = np.append(mean_squared_errors, (sum(outputs-predicted_outputs)**2)/inputs.shape[0])
                mse_graph.plot([i for i in range(1, len(mean_squared_errors)+1)], mean_squared_errors, color='black')

                if inputs.shape[1] == 1:
                    x = np.arange(normalized_inputs[0]-1, normalized_inputs[-1]+1, 0.1)
                    model_graph.plot(x,np.array([self.sigmoid(value) for value in x]), color = 'black', label="Sigmoid curve")
                    model_graph.set_xlabel("Normalized input")
                    model_graph.set_ylabel("Probability of belonging to the red class")
                else:
                    z = 1 / (1 + np.exp(- model.bias - model.coefficients[0] * x - model.coefficients[1] * y))
                    model_graph.plot_surface(x,y,z, linewidth=0, cmap="jet", alpha=0.5)
                    model_graph.scatter(normalized_inputs[:,0], normalized_inputs[:,1], outputs, c=outputs, cmap="coolwarm", s=30)
                camera.snap()

            self.bias -= learning_rate * -2 * sum((outputs - predicted_outputs) * predicted_outputs * (1 - predicted_outputs))
            dloss_dcoefficients = np.zeros(self.coefficients.size)
            for index in range(self.coefficients.size):
                dloss_dcoefficients[index] += -2 * sum((outputs - predicted_outputs) * predicted_outputs * (1 - predicted_outputs) * normalized_inputs[:,index])
            self.coefficients -= learning_rate * dloss_dcoefficients
        
        if display:
            anim = camera.animate()
            anim.save("animation_3d30fps_logistic_regression.gif", fps=30)
        
    def test(self, inputs, expected_outputs):

        '''Tests the model, using the given inputs and comparing them to the given expected outputs. This method will
        print the obtained outputs, as well as the model's accuracy.'''
        
        inputs = self.normalize(inputs)
        outputs = np.array([self.predict(input) for input in inputs])

        count = 0
        for i in range(inputs.shape[0]):
            if outputs[i] == expected_outputs[i]:
                count += 1
        
        accuracy = count / inputs.shape[0]
        print("Outputs:", outputs,"\nAccuracy:", accuracy)

model = LogisticRegression()

model.train(np.array([[1,0],[2,5],[3,-1],[4,7],[5,8],[6,4],[7,8]]), np.array([1,1,1,0,0,0,0]), 0.1, 100, True)
#model.test(np.array([[1],[2],[3],[4],[5],[6],[7],[8]]), np.array([0,0,0,1,1,1,1,1]))

#Nota: se os dados de treino estiverem perfeitamente balanceados, a derivada do loss será 0