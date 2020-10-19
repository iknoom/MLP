#include <iostream>
#include <vector>
#include "MLP.cpp"

MLP and_gate(int n) {
	int data_size, layer_size, input_size, node_size;
	data_size = 1 << n;
	layer_size = 3;
	input_size = n;
	node_size = 3;

	// 학습 데이터 생성 [x : 비트 배열, t : and값]
	std::vector<std::vector<double>> x(data_size);
	std::vector<int> t(data_size, 1);
	for (int i = 0; i < data_size; i++) {
		x[i].resize(n);
		for (int j = 0; j < n; j++)
			x[i][j] = (i & (1 << j)) ? 1 : 0;
		for (int j = 0; j < n; j++)
			t[i] &= (int)x[i][j];
	}

	// 학습 시작
	MLP AND_gate = MLP(layer_size, input_size, node_size);
	int epoch = 0;
	while (true) {
		epoch++;
		double sum_error = 0;
		for (int i = 0; i < data_size; i++) {
			AND_gate.predict(x[i], (double)t[i]);
			AND_gate.learn();
			sum_error += AND_gate.loss();
		}
		if (sum_error < 0.01) break;
	}
	std::cout << "epoch : " << epoch << std::endl;
	return AND_gate;
}

MLP or_gate(int n) {
	int data_size, layer_size, input_size, node_size;
	data_size = 1 << n;
	layer_size = 3;
	input_size = n;
	node_size = 3;

	// 학습 데이터 생성 [x : 비트 배열, t : or값]
	std::vector<std::vector<double>> x(data_size);
	std::vector<int> t(data_size, 0);
	for (int i = 0; i < data_size; i++) {
		x[i].resize(n);
		for (int j = 0; j < n; j++)
			x[i][j] = (i & (1 << j)) ? 1 : 0;
		for (int j = 0; j < n; j++)
			t[i] |= (int)x[i][j];
	}

	// 학습 시작
	MLP OR_gate = MLP(layer_size, input_size, node_size);
	int epoch = 0;
	while (true) {
		epoch++;
		double sum_error = 0;
		for (int i = 0; i < data_size; i++) {
			OR_gate.predict(x[i], (double)t[i]);
			OR_gate.learn();
			sum_error += OR_gate.loss();
		}
		if (sum_error < 0.01) break;
	}
	std::cout << "epoch : " << epoch << std::endl;
	return OR_gate;
}

MLP xor_gate(int n) {
	int data_size, layer_size, input_size, node_size;
	data_size = 1 << n;
	layer_size = 3;
	input_size = n;
	node_size = 3;

	// 학습 데이터 생성 [x : 비트 배열, t : xor값]
	std::vector<std::vector<double>> x(data_size);
	std::vector<int> t(data_size, 0);
	for (int i = 0; i < data_size; i++) {
		x[i].resize(n);
		for (int j = 0; j < n; j++)
			x[i][j] = (i & (1 << j)) ? 1 : 0;
		for (int j = 0; j < n; j++)
			t[i] ^= (int)x[i][j];
	}

	// 학습 시작
	MLP XOR_gate = MLP(layer_size, input_size, node_size);
	int epoch = 0;
	while (true) {
		epoch++;
		double sum_error = 0;
		for (int i = 0; i < data_size; i++) {
			XOR_gate.predict(x[i], (double)t[i]);
			XOR_gate.learn();
			sum_error += XOR_gate.loss();
		}
		if (sum_error < 0.01) break;
	}
	std::cout << "epoch : " << epoch << std::endl;
	return XOR_gate;
}

int main() {
	int bit_size = 2;
	MLP AND_gate = and_gate(bit_size);
	std::cout << "AND GATE :" << '\n';
	std::cout << "0 & 0 = " << AND_gate.predict({ 0, 0 }) << '\n';
	std::cout << "1 & 0 = " << AND_gate.predict({ 1, 0 }) << '\n';
	std::cout << "0 & 1 = " << AND_gate.predict({ 0, 1 }) << '\n';
	std::cout << "1 & 1 = " << AND_gate.predict({ 1, 1 }) << '\n';

	MLP OR_gate = or_gate(bit_size);
	std::cout << "OR GATE :" << '\n';
	std::cout << "0 | 0 = " << OR_gate.predict({ 0, 0 }) << '\n';
	std::cout << "1 | 0 = " << OR_gate.predict({ 1, 0 }) << '\n';
	std::cout << "0 | 1 = " << OR_gate.predict({ 0, 1 }) << '\n';
	std::cout << "1 | 1 = " << OR_gate.predict({ 1, 1 }) << '\n';

	MLP XOR_gate = xor_gate(bit_size);
	std::cout << "XOR GATE :" << '\n';
	std::cout << "0 ^ 0 = " << XOR_gate.predict({ 0, 0 }) << '\n';
	std::cout << "1 ^ 0 = " << XOR_gate.predict({ 1, 0 }) << '\n';
	std::cout << "0 ^ 1 = " << XOR_gate.predict({ 0, 1 }) << '\n';
	std::cout << "1 ^ 1 = " << XOR_gate.predict({ 1, 1 }) << '\n';
}
