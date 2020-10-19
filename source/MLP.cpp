#include <iostream>
#include <vector>
#include <random>
#include <cmath>

class Sigmoid {
private:
	int n;
	std::vector<double> f_net, df;

public:
	Sigmoid() {}
	Sigmoid(int size) : n(size) {
		f_net.resize(n, 0);
		df.resize(n, 0);
	}
	// f(net)을 구하는 함수
	std::vector<double> activate(std::vector<double> net) {
		for (int c = 0; c < n; c++) {
			f_net[c] = 1.0 / (1.0 + exp(-net[c]));
			df[c] = f_net[c] * (1 - f_net[c]);
		}
		return f_net;
	}
	// Sigmoid에서의 역전파 (LeCun's backpropagation)
	std::vector<double> backward(std::vector<double> delta) {
		std::vector<double> next_delta(n);
		for (int i = 0; i < n; i++) {
			next_delta[i] = delta[i] * df[i];
		}
		return next_delta;
	}
};

class Sigmoid_with_loss {
private:
	double f_net, t;

public:
	// f(net)을 구하는 함수
	double activate(double net, double t1) {
		t = t1;
		f_net = 1.0 / (1.0 + exp(-net));
		return f_net;
	}
	// loss까지 계산한 역전파 : delta = -(t - f(net)) * f'(net)
	double backward() {
		double df, delta;
		df = f_net * (1 - f_net);
		delta = -(t - f_net) * df;
		return delta;
	}
	// error를 구하는 함수
	double get_loss() {
		double loss;
		loss = (t - f_net) * (t - f_net) / 2;
		return loss;
	}
};

// 활성화 함수로 sigmoid를 가지는 layer
class Layer {
private:
	int row_n, col_n;
	std::vector<double> x, b;
	std::vector<std::vector<double>> W;
	Sigmoid activation_function;

public:
	Layer() {}
	Layer(int row_size, int col_size) : row_n(row_size), col_n(col_size) {
		// 가중치를 랜덤값으로 초기화
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dis(-100, 100);
		W.resize(row_n);
		for (int r = 0; r < row_n; r++) {
			W[r].resize(col_n);
			for (int c = 0; c < col_n; c++) {
				W[r][c] = (double)dis(gen) / 100.0;
			}
		}
		b.resize(col_n);
		for (int c = 0; c < col_n; c++) {
			b[c] = (double)dis(gen) / 100.0;
		}
		activation_function = Sigmoid(col_n);
	}
	// net을 구하는 함수(w1*x1 + w2*x2 + w3*x3 + ...)
	std::vector<double> evaluate(std::vector<double> x1) {	
		std::vector<double> net(col_n, 0);
		x = x1;
		for (int c = 0; c < col_n; c++) {
			for (int r = 0; r < row_n; r++) {
				net[c] += x1[r] * W[r][c];
			}
			net[c] += b[c];
		}
		return net;
	}
	// f(net)을 구하는 함수
	std::vector<double> activate(std::vector<double> net) {
		std::vector<double> f_net;
		f_net = activation_function.activate(net);
		return f_net;
	}
	// layer에서의 역전파 (LeCun's backpropagation)
	std::vector<double> backward(std::vector<double> delta, double learning_rate) {
		std::vector<double> next_delta(row_n, 0);
		for (int r = 0; r < row_n; r++) {
			for (int c = 0; c < col_n; c++) {
				next_delta[r] += delta[c] * W[r][c];
				W[r][c] -= learning_rate * delta[c] * x[r];
			}
		}
		for (int c = 0; c < col_n; c++) {
			b[c] -= learning_rate * delta[c];
		}
		return next_delta;
	}
	// activation_function의 역전파 값을 불러오는 함수
	std::vector<double> activate_backward(std::vector<double> delta) {
		return activation_function.backward(delta);
	}
};

class MLP {
private:
	int input_n, hidden_n, layer_n;
	double learning_rate, threshold;
	std::vector<Layer> layers;
	Sigmoid_with_loss last_layer;
	
public:
	MLP(int layer_size, int input_size, int hidden_size) {
		layer_n = layer_size;
		input_n = input_size;
		hidden_n = hidden_size;
		learning_rate = 1;
		threshold = 0.5;
		layers.resize(layer_n + 1);
		// input layer
		layers[0] = Layer(input_n, hidden_n);
		// hidden layer
		for (int i = 1; i < layer_n; i++) {
			layers[i] = Layer(hidden_n, hidden_n);
		}
		// output layer
		layers[layer_n] = Layer(hidden_n, 1);
		last_layer = Sigmoid_with_loss();
	}
	// forward sweeping
	int predict(std::vector<double> x, double t = -1.0) {
		double net, f_net;
		for (int i = 0; i < layer_n; i++) {
			x = layers[i].evaluate(x);
			x = layers[i].activate(x);
		}
		net = layers[layer_n].evaluate(x).front();
		f_net = last_layer.activate(net, t);
		return f_net > threshold ? 1 : 0;
	}
	// backward sweeping
	void learn() {
		std::vector<double> delta;
		delta.resize(1, last_layer.backward());
		delta = layers[layer_n].backward(delta, learning_rate);
		for (int i = layer_n - 1; i >= 0; i--) {
			delta = layers[i].activate_backward(delta);
			delta = layers[i].backward(delta, learning_rate);
		}
	}
	// error를 구하는 함수
	double loss() {
		return last_layer.get_loss();
	}
};