#include "dataBase.hpp"
#include "ProgressBar.hpp"
#include <fstream>
#include <iostream>

namespace nn::model {
DataBase::DataBase(const TrainingConfig &_config) : config(_config) {
	std::random_device rd;
	rng = std::mt19937(std::random_device{}());
}

int DataBase::readLine(const std::string &line, TrainSample &sample) {
	sample.pre.index = std::numeric_limits<size_t>::max();
	size_t input_count = 0;

	const char *ptr = line.c_str();
	const char *end = ptr + line.size();

	while (ptr < end) {
		// Skip whitespace
		while (ptr < end && (*ptr == ' ' || *ptr == '\t' || *ptr == '\r' || *ptr == '\n')) {
			++ptr;
		}
		if (ptr >= end)
			break;

		if (*ptr == 'p') {
			++ptr;
			size_t idx = 0;
			while (ptr < end && *ptr >= '0' && *ptr <= '9') {
				idx = idx * 10 + (*ptr - '0');
				++ptr;
			}
			sample.pre.index = idx;
		} else if ((*ptr >= '0' && *ptr <= '9') || *ptr == '-' || *ptr == '+') {
			// Parse number
			double value = 0.0;
			bool negative = false;
			if (*ptr == '-') {
				negative = true;
				++ptr;
			} else if (*ptr == '+') {
				++ptr;
			}

			while (ptr < end && *ptr >= '0' && *ptr <= '9') {
				value = value * 10.0 + (*ptr - '0');
				++ptr;
			}

			if (ptr < end && *ptr == '.') {
				++ptr;
				double frac = 0.0, factor = 0.1;
				while (ptr < end && *ptr >= '0' && *ptr <= '9') {
					frac += (*ptr - '0') * factor;
					factor *= 0.1;
					++ptr;
				}
				value += frac;
			}

			value = negative ? -value : value;

			if (input_count < tempData.size()) {
				tempData[input_count++] = value;
			} else {
				// Skip remaining number if overflow
				while (ptr < end && *ptr != ' ' && *ptr != '\t' && *ptr != '\r' && *ptr != '\n') {
					++ptr;
				}
			}
		} else {
			// Skip unknown token
			while (ptr < end && *ptr != ' ' && *ptr != '\t' && *ptr != '\r' && *ptr != '\n') {
				++ptr;
			}
		}
	}

	if (input_count != tempData.size()) {
		throw std::runtime_error("Expected " + std::to_string(tempData.size()) +
		                         " inputs, got " + std::to_string(input_count));
	}

	if (sample.pre.index == std::numeric_limits<size_t>::max()) {
		throw std::runtime_error("Missing pre.index in line: " + line);
	}

	sample.input = tempData;
	return 0;
}

databaseStatus DataBase::getDataBaseStatus(const std::string &line) {
	std::istringstream iss(line);

	databaseStatus status{0, 0, 0};

	iss >> status.dataBaseSize;
	iss >> status.sampleInputSize;

	tempData.resize(status.sampleInputSize);

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

	samples.status = getDataBaseStatus(line);
	size_t i = samples.size();
	samples.samples.resize(samples.size() + samples.status.dataBaseSize,
	                       {samples.status.sampleOutputSize,
	                        samples.status.sampleInputSize});

	const auto HEADER = LOADING_DB_MESSAGE + db_filename +
	                    DATABASE_FILE_EXETENTION;
	ProgressBar bar(samples.size(), HEADER);

	while (getline(file, line)) {
		if (line.empty() || line[0] == '-') {
			continue;
		}
		try {
			readLine(line, samples.samples[i]);
			++i;
			++bar;
			bar.printBar();
		} catch (const std::exception &e) {
			std::cerr << "Skipping invalid line " << i << ": " << e.what() << "\n";
			continue;
		}
	}

	if (samples.samples.size() > i) {
		samples.samples.resize(i);
		samples.status.dataBaseSize = samples.samples.size();
	}

	bar.endPrint();
	file.close();
	return 0;
}

void DataBase::load(const std::vector<std::string> &db_filenames) {
	std::vector<std::string> errors;

	for (const auto &name : db_filenames) {
		try {
			loadData(name);
		} catch (const std::exception &e) {
			errors.emplace_back("Failed to load " + name + ": " + e.what());
		}
	}

	if (!errors.empty()) {
		std::ostringstream oss;
		oss << "Database load encountered " << errors.size() << " error(s):\n";
		for (const auto &msg : errors) {
			oss << "  - " << msg << "\n";
		}
		throw std::runtime_error(oss.str());
	}

	samples.samples.shrink_to_fit();
	shuffled_indices.resize(samples.size());
	std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
	generateBatches();

	std::cout << "Loaded " << samples.size() << " samples." << std::endl;
}

void DataBase::generateBatches() {
	shuffle(shuffled_indices.begin(), shuffled_indices.end(), rng);

	batches.clear();
	size_t num_batches_expected = (samples.size() + config.getBatchSize() - 1) / config.getBatchSize();
	batches.reserve(num_batches_expected);

	for (size_t i = 0; i < samples.size(); i += config.getBatchSize()) {
		size_t current_batch_actual_size = std::min(config.getBatchSize(), samples.size() - i);

		if (current_batch_actual_size == 0) {
			break;
		}

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
