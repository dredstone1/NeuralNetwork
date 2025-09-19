#include "dataBase.hpp"
#include "ProgressBar.hpp"
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

namespace nn::model {

inline void skipWhitespace(const char *&ptr, const char *end) {
	while (ptr < end && (*ptr == ' ' || *ptr == '\t' || *ptr == '\r' || *ptr == '\n')) {
		++ptr;
	}
}

inline size_t parseIndex(const char *&ptr, const char *end) {
	size_t idx = 0;
	while (ptr < end && *ptr >= '0' && *ptr <= '9') {
		idx = idx * 10 + (*ptr - '0');
		++ptr;
	}
	return idx;
}

inline double parseNumber(const char *&ptr, const char *end) {
	double value = 0.0;
	bool negative = false;

	if (*ptr == '-') {
		negative = true;
		++ptr;
	} else if (*ptr == '+') {
		++ptr;
	}

	// Integer part
	while (ptr < end && *ptr >= '0' && *ptr <= '9') {
		value = value * 10.0 + (*ptr - '0');
		++ptr;
	}

	// Fraction part
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

	return negative ? -value : value;
}

inline void skipToken(const char *&ptr, const char *end) {
	while (ptr < end && *ptr != ' ' && *ptr != '\t' && *ptr != '\r' && *ptr != '\n') {
		++ptr;
	}
}

int DataBase::readLine(const std::string &line, TrainSample &sample) {
	sample.pre.index = std::numeric_limits<size_t>::max();
	size_t input_count = 0;

	const char *ptr = line.c_str();
	const char *end = ptr + line.size();

	while (ptr < end) {
		skipWhitespace(ptr, end);
		switch (*ptr) {
		case 'p':
			++ptr;
			sample.pre.index = parseIndex(ptr, end);
			break;

		case 'w':
			++ptr;
			sample.weight = parseNumber(ptr, end);
			break;

		default:
			if ((*ptr >= '0' && *ptr <= '9') || *ptr == '-' || *ptr == '+') {
				double value = parseNumber(ptr, end);

				if (input_count < tempData.size()) {
					tempData[input_count++] = value;
					break;
				}
			}

			skipToken(ptr, end);
			break;
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

	iss >> status.dbSize;
	iss >> status.sampleInputSize;

	tempData.resize(status.sampleInputSize);

	return status;
}

int DataBase::loadData(const std::string &fileNames) {
	std::ifstream file(fileNames + DATABASE_FILE_EXETENTION);
	if (!file.is_open()) {
		std::cout << FILE_NOT_FOUND_MESSAGE << fileNames << std::endl;
		return 1;
	}

	std::string line;
	getline(file, line);

	databaseStatus s = getDataBaseStatus(line);
	samples.status.sampleInputSize = s.sampleInputSize;
	samples.status.sampleOutputSize = s.sampleOutputSize;

	size_t i = samples.status.dbSize;
	samples.samples.resize(i + s.dbSize,
	                       {samples.status.sampleOutputSize,
	                        samples.status.sampleInputSize});

	const auto HEADER = LOADING_DB_MESSAGE + fileNames +
	                    DATABASE_FILE_EXETENTION;
	ProgressBar bar(s.dbSize, HEADER);

	while (i < samples.status.dbSize + s.dbSize && getline(file, line)) {
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

	samples.status.dbSize = i;

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

	samples.samples.resize(samples.status.dbSize);
	std::cout << "Loaded " << samples.status.dbSize << " samples." << std::endl;
}

TrainSample DataBase::getSample(const size_t i) {
	TrainSample newSample;
	newSample.pre = samples.samples[i].pre;
	newSample.weight = samples.samples[i].weight;

	newSample.input.shape = samples.samples[i].input.shape;
	newSample.input.strides = samples.samples[i].input.strides;

	if (nn::global::Tensor::getGpuState()) {
		newSample.input.gpu_data = samples.samples[i].input.gpu_data;
	} else {
		newSample.input.cpu_data = samples.samples[i].input.cpu_data;
	}
	return newSample;
}

} // namespace nn::model
