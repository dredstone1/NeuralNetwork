#include "dataBase.hpp"
#include "ProgressBar.hpp"
#include "tensor.hpp"
#include <fstream>
#include <iostream>
#include <ostream>
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

	TrainSample new_sample(
	    samples.status.sampleOutputSize,
	    samples.status.sampleInputSize);

	new_sample.pre.index = std::stoull(token);

	std::vector<global::ValueType> data(new_sample.input.numElements());
	for (size_t i = 0; i < samples.status.sampleInputSize; ++i) {
		iss >> token;

		data[i] = std::stod(token);
	}
	new_sample.input = data;

	return new_sample;
}

databaseStatus DataBase::getDataBaseStatus(const std::string &line) {
	std::istringstream iss(line);

	databaseStatus status{0, 0, 0};

	iss >> status.dataBaseSize;
	iss >> status.sampleInputSize;

	return status;
}

int DataBase::loadData(const std::string &db_filename) {
	std::ifstream file(db_filename + DATABASE_FILE_EXETENTION);

	if (!file.is_open()) {
		std::cout << FILE_NOT_FOUND_MESSAGE << db_filename << std::endl;
		return 1;
	}

	std::string line;
	getline(file, line);

	size_t tempSize = samples.status.dataBaseSize;

	samples.status = getDataBaseStatus(line);
	getline(file, line);

	samples.samples.reserve(samples.size() + samples.status.dataBaseSize);

	ProgressBar bar(samples.size() - tempSize, LOADING_DB_MESSAGE + db_filename);

	while (getline(file, line)) {
		if (line.empty() ||
		    line.find_first_not_of(" \t\n\v\f\r") == std::string::npos) {
			continue;
		}

		TrainSample new_sample = readLine(line);
		if (new_sample.input.numElements() == 0) {
			continue;
		}

		samples.add(new_sample);

		bar++;
		bar.printBar();
	}

	if (samples.samples.capacity() > samples.size()) {
		bar = samples.samples.capacity();
		bar.printBar();

		samples.samples.shrink_to_fit();
		samples.status.dataBaseSize = samples.samples.size();
	}

	file.close();

	shuffled_indices.resize(samples.size());
	iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
	generateBatches();

	return 0;
}

int DataBase::load(const std::string &db_filenames) {
	return loadData(db_filenames);
	std::cout << "Loaded " << samples.size() << " samples." << std::endl;
}

int DataBase::load(const std::vector<std::string> &db_filenames) {
	for (auto name : db_filenames) {
		if (loadData(name)) {
			return 1;
		}
	}
	std::cout << "Loaded " << samples.size() << " samples." << std::endl;
	return 0;
}

void DataBase::generateBatches() {
	shuffle(shuffled_indices.begin(), shuffled_indices.end(), rng);

	batches.clear();
	size_t num_batches_expected = (samples.size() + config.getBatchSize() - 1) / config.getBatchSize();
	batches.reserve(num_batches_expected);

	for (size_t i = 0; i < samples.size(); i += config.getBatchSize()) {
		size_t current_batch_actual_size = std::min(config.getBatchSize(), samples.size() - i);

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
