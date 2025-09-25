#include "model.hpp"
#include "tensor.hpp"
#include "tests.hpp"
#include <iostream>

int main() {
	nn::global::Tensor::toCpu();
	nn::model::Model model(tests::appendToBase("config_statistic_test.json"));
	nn::model::DataBase dbt;
	dbt.load({"../tests/data/statistic_db"});
	model.train(dbt, dbt);

	while (true) {
		double a, b;
		std::cout << "Enter two numbers (or non-number to quit): ";
		if (!(std::cin >> a >> b))
			break;

		nn::global::Tensor input({2});
		input.setValue(0, a);
		input.setValue(1, b);

		model.runModel(input);

		const auto &out = model.getOut();
		std::cout << "Output: ";
		for (auto v : out)
			std::cout << v << " ";
		std::cout << "\nPrediction: index=" << model.getPrediction().index
		          << " value=" << model.getPrediction().value << "\n\n";
	}
}
