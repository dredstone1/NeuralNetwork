#include "dataBase.hpp"
#include <fstream>
#include <iostream>
#include <memory>

namespace nn {
DataBase::DataBase(TrainingConfig &config_)
    : samples(nullptr),
      config(config_),
      currentBatch(0) {
	std::random_device rd;
	rng = std::mt19937(rd());

	load();
	shuffled_indices.resize(samples->size());
	iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
	generete_batches();
}

TrainSample DataBase::read_line(const std::string &line) {
	std::istringstream iss(line);

	std::string token;
	iss >> token;
	size_t best_next_move = std::stoi(token);

	TrainSample new_sample({best_next_move, 1.f}, samples->sInputSize);

	for (int i = 0; i < samples->sInputSize; ++i) {
		iss >> token;

		new_sample.input[i] = std::stod(token);
	}

	return new_sample;
}

void DataBase::getDataBaseStatus(const std::string &line) {
	std::istringstream iss(line);

	int dataBaseSize = 0;
	int sampleSize = 0;

	iss >> dataBaseSize;
	iss >> sampleSize;

	samples = std::make_unique<Samples>(sampleSize, dataBaseSize);
}

int DataBase::load() {
	std::ifstream file(config.db_filename + ".txt");
	if (!file.is_open()) {
		std::cout << "File not found: " << config.db_filename << std::endl;
		return 1;
	}

	std::string line;
	getline(file, line);
	getDataBaseStatus(line);

	while (getline(file, line)) {
		if (line.empty() || line.find_first_not_of(" \t\n\v\f\r") == std::string::npos) {
			continue;
		}
		samples->add(read_line(line));
	}

	if (samples->samples.capacity() > samples->size()) {
		samples->samples.shrink_to_fit();
	}

	std::cout << "Loaded " << (samples ? samples->size() : 0) << " samples." << std::endl;
	file.close();

	return 0;
}

void DataBase::generete_batches() {
	shuffle(shuffled_indices.begin(), shuffled_indices.end(), rng);

	batches.clear();
	size_t num_batches_expected = (samples->size() + config.batch_size - 1) / config.batch_size;
	batches.reserve(num_batches_expected);

	for (size_t i = 0; i < samples->size(); i += config.batch_size) {
		size_t current_batch_actual_size = std::min((size_t)config.batch_size, samples->size() - i);

		if (current_batch_actual_size == 0)
			break;

		batches.emplace_back(current_batch_actual_size);
		Batch &new_batch = batches.back();

		for (size_t j = 0; j < current_batch_actual_size; ++j) {
			int sample_original_index = shuffled_indices[i + j];
			new_batch.samples[j] = &samples->samples[sample_original_index];
		}
	}
}

Batch &DataBase::get_Batch() {
	if (batches.empty() || currentBatch >= batches.size()) {
		generete_batches();
		currentBatch = 0;
	}

	return batches.at(currentBatch++);
}
} // namespace nn
