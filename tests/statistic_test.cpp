#include "model.hpp"
#include "tensor.hpp"
#include "tests.hpp"
#include <iostream>

int main() {
	// Force CPU mode for consistency
	nn::global::Tensor::toCpu();

	// Load model from config file
	std::string config_FN = "config_statistic_test.json";
	nn::model::Model model(tests::appendToBase(config_FN));
	std::vector<std::string> files{"../tests/data/statistic_db"};
	nn::model::DataBase dbt;
	dbt.load(files);
	model.train(dbt, dbt);

	// Create a dummy input tensor with size 10
	size_t input_size = 10;
	nn::global::Tensor input({input_size}, 0.0);

	// Fill with some values
	for (size_t i = 0; i < input_size; ++i) {
		input.setValue(i, (i % 2 == 0) ? 1.0 : 0.5);
	}

	// Print input
	std::cout << "Input tensor: ";
	for (size_t i = 0; i < input.numElements(); ++i) {
		std::cout << input.getValue(i) << " ";
	}
	std::cout << "\n";

	// Run model
	model.runModel(input);

	// Print output tensor
	const auto &output = model.getOut();
	std::cout << "Output tensor: ";
	for (size_t i = 0; i < output.size(); ++i) {
		std::cout << output[i] << " ";
	}
	std::cout << "\n";

	// Print prediction
	auto pred = model.getPrediction();
	std::cout << "Prediction: index=" << pred.index
	          << " value=" << pred.value << "\n";

	return 0;
}
