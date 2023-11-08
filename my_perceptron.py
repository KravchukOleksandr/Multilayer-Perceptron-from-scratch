import numpy as np


class My_Perceptron:
    def __init__(self, n_neurons, act_func, input_len):
        self.n_layers = len(n_neurons)
        self.n_neurons = n_neurons.copy()
        self.input_len = input_len
        self.output_len = n_neurons[-1]
        self.act_func = act_func

    
    def __initialize_net(self):
        self.b = []
        self.a = []
        self.z = []
        self.dc_da = []
        self.w = []
        for k in range(self.n_layers):
            self.a.append(np.zeros(self.n_neurons[k]))
        self.z = self.a.copy()
        self.dc_da = self.a.copy()
        self.b = self.a.copy()
        self.w.append(My_Perceptron.rand_weight(self.input_len, self.n_neurons[0]))
        for k in range(1, self.n_layers):
            self.w.append(My_Perceptron.rand_weight(self.n_neurons[k - 1], self.n_neurons[k]))


    def fit(self, X, y, epochs, l_rate):
        X, y = X.copy(), y.copy()
        self.__initialize_net()
        y = y.reshape(-1, self.output_len)
        n_samples = X.shape[0]
        for _ in range(epochs):
            for n in range(n_samples):
                self.__forward_prop(X[n])
                self.__back_prop(X[n], y[n], l_rate)
        return self


    def predict(self, X_test):
        n_samples = X_test.shape[0]
        y_pred = np.zeros([n_samples, self.output_len])
        for n in range(n_samples):
            x_sample = X_test[n]
            for k in range(self.n_layers):
                x_sample = My_Perceptron.func(self.w[k].T @ x_sample + self.b[k],
                                              self.act_func[k])
            y_pred[n] = x_sample
        return y_pred


    def __forward_prop(self, X_n):
        self.z[0] = self.w[0].T @ X_n + self.b[0]
        self.a[0] = My_Perceptron.func(self.z[0], self.act_func[0])
        for k in range(1, self.n_layers):
            self.z[k] = self.w[k].T @ self.a[k - 1] + self.b[k]
            self.a[k] = My_Perceptron.func(self.z[k], self.act_func[k])


    def __back_prop(self, X_n, y_n, l_rate):
        self.dc_da[-1]= (self.a[-1] - y_n) * My_Perceptron.deriv(self.z[-1], self.act_func[-1])
        for k in range(self.n_layers - 2, -1, -1):
            self.dc_da[k] = self.w[k + 1] @ self.dc_da[k + 1] * My_Perceptron.deriv(self.z[k], 
                                                                                    self.act_func[k])
        
        self.w[0] = self.w[0] - l_rate * (X_n.reshape(-1, 1) @ self.dc_da[0].reshape(1, -1))
        self.b[0] = self.b[0] - l_rate * self.dc_da[0]

        for k in range(1, self.n_layers):
            self.w[k] = self.w[k] - l_rate * (self.a[k - 1].reshape(-1, 1) @ self.dc_da[k].reshape(1, -1))
            self.b[k] = self.b[k] - l_rate * self.dc_da[k]


    @staticmethod
    def func(x, func_name):
        if func_name == 'relu':
            return np.maximum(0, x)
        if func_name == 'leaky_relu':
            return np.maximum(0.05 * x, x)
        if func_name == 'sigmoid':
            x = 1 / (1 + np.exp(-x))
            return x
        if func_name == 'default':
            return x
        if func_name == 'softmax':
            C = np.sum(np.exp(x))
            return np.exp(x) / C
        if func_name == 'heaviside':
            return np.where(x >= 0, 1, 0)


    @staticmethod
    def deriv(x, func_name):
        if func_name == 'relu':
            return np.where(x >= 0, 1, 0)
        if func_name == 'leaky_relu':
            return np.where(x >= 0, 1, 0.05)
        if func_name == 'sigmoid':
            x = 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))
            return x
        if func_name == 'default':
            return 1
        if func_name == 'softmax':
            f = My_Perceptron.func(x, 'softmax')
            return f * (1 - f)
        if func_name == 'heaviside':
            return np.ones(x.shape)
        
    
    @staticmethod
    def rand_weight(f_in, f_out):
        w_size = (f_in, f_out)
        C = np.sqrt(6) / np.sqrt(f_in + f_out)
        return np.random.uniform(-C, C, w_size)
