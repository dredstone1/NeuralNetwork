#include "dataBase.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace nn::model {
DataBase::DataBase(const TrainingConfig &_config) : config(_config) {
	std::random_device rd;
	rng = std::mt19937(std::random_device{}());
}

TrainSample DataBase::readLine(const std::string &line) {
	std::istringstream iss(line);

	std::string token;
	iss >> token;
	if (token == ("--")) {
		return TrainSample();
	}

	TrainSample new_sample(samples.sOutputSize, samples.sInputSize);

	new_sample.pre.index = std::stoull(token);

	for (int i = 0; i < samples.sInputSize; ++i) {
		iss >> token;

		new_sample.input[i] = std::stod(token);
	}

	return new_sample;
}

void DataBase::getDataBaseStatus(const std::string &line) {
	std::istringstream iss(line);

	int dataBaseSize = 0, sampleInputSize = 0;

	iss >> dataBaseSize;
	iss >> sampleInputSize;

	samples.sInputSize = sampleInputSize;

	samples.samples.reserve(samples.size() + dataBaseSize);
}

int DataBase::load(const std::string &db_filename) {
	std::ifstream file(db_filename + DATABASE_FILE_EXETENTION);
	if (!file.is_open()) {
		std::cout << "File not found: " << db_filename << std::endl;
		return 1;
	}

	std::cout << "Start loading data base" << std::endl;

	std::string line;
	getline(file, line);
	getDataBaseStatus(line);

	while (getline(file, line)) {
		if (line.empty() || line.find_first_not_of(" \t\n\v\f\r") == std::string::npos) {
			continue;
		}
		TrainSample new_sample = readLine(line);
		if (new_sample.input.size() == 0)
			continue;

		samples.add(new_sample);
	}

	if (samples.samples.capacity() > samples.size()) {
		samples.samples.shrink_to_fit();
	}

	std::cout << "Loaded " << samples.size() << " samples." << "\n";
	if (config.getBatchSize() > samples.size()) {
		std::cout << "batch size too big" << "\n";
		return 1;
	}
	file.close();

	shuffled_indices.resize(samples.size());
	iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
	generateBatches();

	return 0;
}

int DataBase::load(const std::vector<std::string> &db_filenames) {
	for (auto name : db_filenames) {
		int error = load(name);

		if (error) {
			return error;
		}
	}

	return 0;
}

void DataBase::generateBatches() {
	shuffle(shuffled_indices.begin(), shuffled_indices.end(), rng);

	batches.clear();
	size_t num_batches_expected = (samples.size() + config.getBatchSize() - 1) / config.getBatchSize();
	batches.reserve(num_batches_expected);

	for (size_t i = 0; i < samples.size(); i += config.getBatchSize()) {
		size_t current_batch_actual_size = std::min((size_t)config.getBatchSize(), samples.size() - i);

		if (current_batch_actual_size == 0)
			break;

		batches.emplace_back(current_batch_actual_size);
		Batch &new_batch = batches.back();

		for (size_t j = 0; j < current_batch_actual_size; ++j) {
			int sample_original_index = shuffled_indices[i + j];
			new_batch.samples[j] = &samples.samples[sample_original_index];
		}
	}
}

Batch &DataBase::getBatch() {
	if (batches.empty() || currentBatch >= batches.size()) {
		generateBatches();
		currentBatch = 0;
	}

	return batches.at(currentBatch++);
}
} // namespace nn::model
