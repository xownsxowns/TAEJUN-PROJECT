import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def build_model(x, hidden_nodes, output_dim=1):
    model = {}
    input_dim = x.shape[1] # input dimension
    model['w1'] = np.random.randn(input_dim, hidden_nodes)
    model['b1'] = np.zeros((1, hidden_nodes))
    model['w2'] = np.random.randn(hidden_nodes, output_dim)
    model['b2'] = np.zeros((1, output_dim))
    return model

def feed_forward(model, x):
    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
    # forward propagation
    z1 = x.dot(w1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(w2) + b2
    output  = sigmoid(z2)
    return z1, a1, z2, output

def backprop(x,y,model,z1,a1,z2,output):
    # using delta rule (gradient)
    delta3 = output
    delta3 = delta3 - y
    dw2 = (a1.T).dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = delta3.dot(model['w2'].T) * sigmoid_derivative(a1)
    dw1 = np.dot(x.T, delta2)
    db1 = np.sum(delta2, axis=0)
    return dw1, dw2, db1, db2

def train(model, x, y, num_passes=10000, learning_rate = 0.1, iteration=10000):
    for i in range(iteration):
        z1, a1, z2, output = feed_forward(model, x)
        dw1, dw2, db1, db2 = backprop(x,y,model,z1,a1,z2,output)
        model['w1'] -= learning_rate * dw1
        model['b1'] -= learning_rate * db1
        model['w2'] -= learning_rate * dw2
        model['b2'] -= learning_rate * db2
        print("predicted output: \n" + str(output))
        print("loss: \n" + str(np.mean(np.square(y - output))))
    return model

def main():
    x = np.array(([0, 0], [1, 0], [0, 1], [1, 1]), dtype=int)
    y = np.array(([0], [1], [1], [0]), dtype=int)
    num_examples = len(x)
    nn_input_dim = 2
    nn_output_dim = 1
    learning_rate = 0.1
    model = build_model(x, 2, 1)
    model = train(model, x, y, learning_rate=learning_rate)
    output = feed_forward(model, x)

if __name__ == "__main__":
    main()
