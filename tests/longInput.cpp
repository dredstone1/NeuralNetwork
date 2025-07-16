#include "tests.hpp"
#include <model.hpp>

int main(int argc, char *argv[]) {
	std::string config_FN = tests::appendToBase("config-long_test.json");

    nn::model::Model model(config_FN);
    while (true) {}

	return 0;
}
