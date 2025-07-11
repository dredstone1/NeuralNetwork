#include "tests.hpp"
#include <AiModel.hpp>

int main(int argc, char *argv[]) {
	std::string config_FN = tests::appendToBase("config-long_test.json");

	nn::AiModel model(config_FN);
    while (true) {}

	return 0;
}
