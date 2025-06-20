#include "tests.hpp"
#include <AiModel.hpp>
#include <iostream>

int int_to_binaray(int num) {
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

int largest_num_from_bit_amount(int bit_amount) {
	return (1 << bit_amount) - 1;
}

void print_database(int actual_size, int input_size, int database_size) {
	int max_num = largest_num_from_bit_amount(actual_size);

	int count = 0;
	for (int num = 0; num <= max_num && count < database_size; ++num) {
		for (int shift = 0; shift <= input_size - actual_size && count < database_size; ++shift) {
			std::cout << std::setw(2) << num << " ";

			for (int i = 0; i < input_size; i++) {
				int bit_index = i - shift;

				if (bit_index < 0 || bit_index >= actual_size) {
					std::cout << 0.1 << " ";
				} else {
					int bit = (num >> (actual_size - 1 - bit_index)) & 1;

					std::cout << (bit ? 1 : 0.5) << " ";
				}
			}
			std::cout << std::endl;
			++count;
		}
	}
}

void printVector(const nn::global::ParamMetrix &vec) {
	for (const auto &elem : vec) {
		std::cout << elem << ' ';
	}

	std::cout << '\n';
}

int main(int argc, char *argv[]) {
	int input_size = 10;
	// print_database(4, input_size, 1000);

	std::string config_FN = tests::appendToBase("config-binary_test.json");

	nn::AiModel model(config_FN);


	// model.train();

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

		int binary = int_to_binaray(num1);

		std::cout << "binary: " << binary << std::endl;

		nn::global::ParamMetrix input(input_size, 0.1);

		std::cout << "Enter an integer 2: ";
		std::getline(std::cin, str_num);
		if (!isNumber(str_num)) {
			std::cout << str_num << " is not a number, please enter a valid integer" << std::endl;
			continue;
		}
		num2 = std::stoi(str_num);

		for (size_t i = 4 + num2; i > num2; i--) {
			input[i - 1] = bit_by_index(num1, 4 - i+num2);
			if (input[i - 1] == 0) {
				input[i - 1] = 0.5;
			}
		}

		printVector(input);
		model.runModel(input);
		printf("prediction: %d, %f\n", model.getPrediction().index, model.getPrediction().value);
	}

	return 0;
}
