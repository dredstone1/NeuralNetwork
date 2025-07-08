#include "tests.hpp"
#include <AiModel.hpp>
#include <iostream>

int int_to_binary(int num) {
	int binary = 0;
	int place = 1;

	while (num > 0) {
		int bit = num % 2;
		binary += bit * place;
		place *= 10;
		num /= 2;
	}

	return binary;
}

int bit_by_index(int binary, int index) {
	return (binary >> index) & 1;
}

bool isNumber(const std::string &s) {
	if (s.empty())
		return false;

	for (char c : s) {
		if (!std::isdigit(c))
			return false;
	}

	return true;
}

void printVector(const nn::global::ParamMetrix &vec) {
	for (const auto &elem : vec) {
		std::cout << elem << ' ';
	}

	std::cout << '\n';
}

int main(int argc, char *argv[]) {
	std::string config_FN = tests::appendToBase("config-statics_test.json");

	nn::AiModel model(config_FN);

	model.train();

	int num1 = 0, num2 = 0;
	std::string str_num;
	while (num1 != -1) {
		std::cout << "Enter an integer 1: ";
		std::getline(std::cin, str_num);
		if (!isNumber(str_num)) {
			std::cout << str_num << " is not a number, please enter a valid integer" << std::endl;
			continue;
		}
		num1 = std::stoi(str_num);

		if (num1 == -1)
			break;

		int binary = int_to_binary(num1);

		std::cout << "binary: " << binary << std::endl;

		nn::global::ParamMetrix input(2, 0);

		for (size_t i = 2; i > 0; i--) {
			input[i - 1] = bit_by_index(num1, 2 - i);
		}

		printVector(input);
		model.runModel(input);
		printf("prediction: %d, %f\n", model.getPrediction().index, model.getPrediction().value);
	}

	return 0;
}
