#include "dataBase.hpp"
#include "ProgressBar.hpp"
#include <cctype>
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

	auto parse_double = [](const char *&p, const char *end) -> double {
		double value = 0.0;
		bool negative = false;
		if (*p == '-') {
			negative = true;
			++p;
		} else if (*p == '+') {
			++p;
		}

		while (p < end && *p >= '0' && *p <= '9') {
			value = value * 10.0 + (*p - '0');
			++p;
		}

		if (p < end && *p == '.') {
			++p;
			double frac = 0.0, factor = 0.1;
			while (p < end && *p >= '0' && *p <= '9') {
				frac += (*p - '0') * factor;
				factor *= 0.1;
				++p;
			}
			value += frac;
		}

		return negative ? -value : value;
	};

	while (ptr < end) {
		while (ptr < end && (*ptr == ' ' || *ptr == '\t' || *ptr == '\r' ||
		                     *ptr == '\n')) {
			++ptr;
		}
		if (ptr >= end) {
			break;
		}

		if (*ptr == 'p') {
			++ptr;
			size_t idx = 0;

			while (ptr < end && *ptr >= '0' && *ptr <= '9') {
				idx = idx * 10 + (*ptr - '0');
				++ptr;
			}
			sample.pre.index = idx;
		} else if ((*ptr >= '0' && *ptr <= '9') || *ptr == '-' || *ptr == '+') {
			if (input_count < tempData.size()) {
				tempData[input_count++] = parse_double(ptr, end);
			} else {
				while (ptr < end && *ptr != ' ' && *ptr != '\t' &&
				       *ptr != '\r' && *ptr != '\n') {
					++ptr;
				}
			}
		} else {
			while (ptr < end && *ptr != ' ' && *ptr != '\t' && *ptr != '\r' &&
			       *ptr != '\n') {
				++ptr;
			}
		}
	}

	if (input_count != tempData.size()) {
		std::cerr << "Error: expected " << tempData.size() << " inputs, got "
		          << input_count << "\n";
		return 1;
	}

	if (sample.pre.index == std::numeric_limits<size_t>::max()) {
		std::cerr << "Error: pre.index not set in line: " << line << "\n";
		return 1;
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
	auto start = std::chrono::high_resolution_clock::now();
	std::ifstream file(db_filename + DATABASE_FILE_EXETENTION);

	if (!file.is_open()) {
		std::cout << FILE_NOT_FOUND_MESSAGE << db_filename << std::endl;
		return 1;
	}

	std::string line;
	getline(file, line);

	samples.status = getDataBaseStatus(line);
	samples.samples.resize(samples.size() + samples.status.dataBaseSize,
	                       {samples.status.sampleOutputSize,
	                        samples.status.sampleInputSize});

	ProgressBar bar(samples.size(), LOADING_DB_MESSAGE + db_filename + DATABASE_FILE_EXETENTION);

	size_t i = 0;
	while (getline(file, line)) {
		if (line.empty() || line[0] == '-' || readLine(line, samples.samples[i])) {
			continue;
		}
		++i;
		bar++;
		bar.printBar();
	}

	if (samples.samples.capacity() > samples.size()) {
		samples.samples.shrink_to_fit();
		samples.status.dataBaseSize = samples.samples.size();
	}
	bar.endPrint();

	file.close();

	shuffled_indices.resize(samples.size());
	iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
	generateBatches();

	auto end = std::chrono::high_resolution_clock::now();

	// Calculate duration
	std::chrono::duration<double> elapsed = end - start;

	std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
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
