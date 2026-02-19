import numpy as np

class Perceptron:
    def __init__(self,input_size, learning_rate=0.01):
        self.weight = np.random.randn(input_size)*0.01
        self.bias = 0.0
        self.learning_rate = learning_rate

    def predict(self,x):

        z = np.dot(x,self.weight)+self.bias
        return self.activation(z)
    
    def activation(self,z):
        return 1 if z>=0 else 0
    
    def train(self, X, y, epochs=10):

        for epoch in range(epochs):
            errors = 0

            for i in range(len(X)):
                 
                x = X[i]
                y_act = y[i]

                y_pred = self.predict(x)
                error = y_act-y_pred

                if error!=0:
                    self.weight += self.learning_rate*error*x
                    self.bias += self.learning_rate*error
                    errors+=1
            print(f"epoch : {epoch+1}/{epochs} - Error : {errors}")

            if error==0:
                print("****Completed****")
                break

if __name__=='__main__':
    print("=====Training=====")

    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])

    model = Perceptron(input_size=2,learning_rate=0.1)

    print("Initial weight:", model.weight)
    print("Initial bias:", model.bias)

    model.train(X,y,epochs=10)

    print(f"\nFinal weight: {model.weight}")
    print(f"Final bias: {model.bias}")
