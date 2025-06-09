#include <AiModel.hpp>
#include <iostream>
#include <trainer.hpp>

enum mode {
	load = 0,   // 0001
	save = 1,   // 0010
	train = 2,  // 0100
	visual = 4, // 1000
};

bool isNumber(const std::string &s) {
	if (s.empty())
		return false;
	for (char c : s) {
		if (!std::isdigit(c))
			return false;
	}
	return true;
}

int main(int argc, char *argv[]) {
	if (argc > 1) {
		int mods = 0;
		char *arg = argv[1];
		while (*(arg) != '\0') {
			switch (*arg) {
			case 'l':
				mods |= load;
				break;
			case 's':
				mods |= save;
				break;
			case 't':
				mods |= train;
				break;
			case 'v':
				mods |= visual;
				break;
			}

			arg++;
		}
		std::string config_FN = "config.json";

		nn::AiModel model(config_FN, mods & visual);

		if (mods & train) {
            nn::training::Trainer trainer(model);

			trainer.train();
		}

		int num1 = 0, num2 = 0;
		std::string str_num;
		while (num2 != 5) {
			std::cout << "Enter an integer 1: ";
			std::getline(std::cin, str_num);
			if (!isNumber(str_num)) {
				std::cout << str_num << " is not a number, please enter a valid integer" << std::endl;
				continue;
			}
			num1 = std::stoi(str_num);

			if (num1 == 5)
				break;

			std::cout << "Enter an integer 2: ";
			std::getline(std::cin, str_num);
			if (!isNumber(str_num)) {
				std::cout << str_num << " is not a number, please enter a valid integer" << std::endl;
				continue;
			}
			num2 = std::stoi(str_num);

			std::vector<double> input(2, 0);
			input[0] = num1;
			input[1] = num2;
			model.runModel(input);
			printf("prediction: %d, %f\n", model.getPrediction().index, model.getPrediction().value);
		}

		return 0;
	}

	std::cout << "Need at least 1 input!" << std::endl;
	return 1;
}
