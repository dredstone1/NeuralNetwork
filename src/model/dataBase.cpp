#include "ProgressBar.hpp"
#include "tensor.hpp"
#include <dataBase.hpp>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

namespace nn::model {

// ============================================================================
// UTILITY FUNCTIONS FOR PARSING
// ============================================================================

/**
 * @brief Skips whitespace characters in a string
 * @param ptr Pointer to current position in string
 * @param end Pointer to end of string
 */
inline void skipWhitespace(const char *&ptr, const char *end) {
	while (ptr < end && (*ptr == ' ' || *ptr == '\t' || *ptr == '\r' || *ptr == '\n')) {
		++ptr; // advance past whitespace
	}
}

/**
 * @brief Parses an integer index from a string
 * @param ptr Pointer to current position in string
 * @param end Pointer to end of string
 * @return Parsed integer value
 */
inline size_t parseIndex(const char *&ptr, const char *end) {
	size_t idx = 0;
	while (ptr < end && *ptr >= '0' && *ptr <= '9') {
		idx = idx * 10 + (*ptr - '0'); // accumulate integer value
		++ptr;                         // move to next character
	}
	return idx;
}

/**
 * @brief Parses a floating-point number from a string
 * @param ptr Pointer to current position in string
 * @param end Pointer to end of string
 * @return Parsed floating-point value
 */
inline double parseNumber(const char *&ptr, const char *end) {
	double value = 0.0;
	bool negative = false;

	if (*ptr == '-') { // check for negative sign
		negative = true;
		++ptr;
	} else if (*ptr == '+') { // skip positive sign
		++ptr;
	}

	// parse integer part
	while (ptr < end && *ptr >= '0' && *ptr <= '9') {
		value = value * 10.0 + (*ptr - '0');
		++ptr;
	}

	// parse fractional part
	if (ptr < end && *ptr == '.') {
		++ptr;
		double frac = 0.0, factor = 0.1;
		while (ptr < end && *ptr >= '0' && *ptr <= '9') {
			frac += (*ptr - '0') * factor; // add digit * factor
			factor *= 0.1;                 // update factor for next digit
			++ptr;
		}
		value += frac; // add fractional part
	}

	return negative ? -value : value; // apply sign
}

/**
 * @brief Skips a token (non-whitespace characters) in a string
 * @param ptr Pointer to current position in string
 * @param end Pointer to end of string
 */
inline void skipToken(const char *&ptr, const char *end) {
	while (ptr < end && *ptr != ' ' && *ptr != '\t' && *ptr != '\r' && *ptr != '\n') {
		++ptr; // skip non-whitespace characters
	}
}

// ============================================================================
// DATABASE IMPLEMENTATION
// ============================================================================

void DataBase::readLine(const std::string &line, TrainSample &sample) {
	sample.index = std::numeric_limits<size_t>::max(); // mark pre.index as invalid
	size_t input_count = 0;
	size_t output_count = 0;

	const char *ptr = line.c_str();
	const char *end = ptr + line.size();

	while (ptr < end) {
		skipWhitespace(ptr, end); // skip spaces/tabs/newlines
		switch (*ptr) {
		case 'p':
			++ptr;
			if (samples.status.outputType == OutType::Statistic) {
				if (input_count < tempDataI.size()) {
					tempDataO[output_count++] = parseNumber(ptr, end);
				}
			} else {
				sample.index = parseIndex(ptr, end);
			}
			break;
		case 'w': // parse sample weight
			++ptr;
			sample.weight = parseNumber(ptr, end);
			break;
		default:
			// parse input value if it looks like a number
			if ((*ptr >= '0' && *ptr <= '9') || *ptr == '-' || *ptr == '+') {
				if (input_count < tempDataI.size()) {
					tempDataI[input_count++] = parseNumber(ptr, end); // store input value
					break;
				}
			}

			skipToken(ptr, end); // skip unrecognized token
			break;
		}
	}

	// ensure the expected number of inputs were read
	if (input_count != tempDataI.size()) {
		throw std::runtime_error("Expected " + std::to_string(tempDataI.size()) +
		                         " inputs, got " + std::to_string(input_count));
	}

	if (samples.status.outputType == OutType::Statistic) {
		sample.out = new global::Tensor({samples.status.sampleOutputSize});
		sample.f_d = true;
		if (sample.out) {
			*sample.out = tempDataO;
		}
	}
	sample.input = tempDataI; // copy tempData to sample input
}

databaseStatus DataBase::getDataBaseStatus(const std::string &line) {
	std::istringstream iss(line);
	databaseStatus status{0, 0, 0, OutType::Classify};

	iss >> status.dbSize;          // read number of samples
	iss >> status.sampleInputSize; // read input vector size
	iss >> status.sampleOutputSize;

	if (status.sampleOutputSize > 0) {
		status.outputType = OutType::Statistic;
		tempDataO.resize(status.sampleOutputSize); // resize temp buffer for parsing
	}

	tempDataI.resize(status.sampleInputSize); // resize temp buffer for parsing

	return status;
}

int DataBase::loadData(const std::string &fileNames) {
	std::ifstream file(fileNames + DATABASE_FILE_EXETENTION);
	if (!file.is_open()) {
		std::cout << FILE_NOT_FOUND_MESSAGE << fileNames << std::endl;
		return 1; // file not found
	}

	std::string line;
	getline(file, line);

	databaseStatus s = getDataBaseStatus(line); // read header info
	samples.status.sampleInputSize = s.sampleInputSize;
	samples.status.outputType = s.outputType;
	samples.status.sampleOutputSize = s.sampleOutputSize;

	size_t i = samples.status.dbSize;
	// resize sample vector to fit new data
	samples.samples.resize(i + s.dbSize, {samples.status.sampleInputSize});

	const auto HEADER = LOADING_DB_MESSAGE + fileNames +
	                    DATABASE_FILE_EXETENTION;
	ProgressBar bar(s.dbSize, HEADER); // initialize progress bar

	while (i < samples.status.dbSize + s.dbSize && getline(file, line)) {
		if (line.empty() || line[0] == '-') {
			continue; // skip empty or comment lines
		}
		try {
			readLine(line, samples.samples[i]); // parse line
			++i;                                // increment sample index
			++bar;                              // increment progress bar
			bar.printBar();                     // display progress
		} catch (const std::exception &e) {
			std::cerr << "Skipping invalid line " << i << ": " << e.what() << "\n";
			continue; // skip invalid line
		}
	}

	samples.status.dbSize = i; // update total sample count

	bar.endPrint(); // finalize progress bar
	file.close();
	return 0;
}

void DataBase::load(const std::vector<std::string> &db_filenames) {
	std::vector<std::string> errors;

	for (const auto &name : db_filenames) {
		try {
			loadData(name); // attempt to load each file
		} catch (const std::exception &e) {
			errors.emplace_back("Failed to load " + name + ": " + e.what());
		}
	}

	if (!errors.empty()) {
		std::ostringstream oss;
		oss << "Database load encountered " << errors.size() << " error(s):\n";
		for (const auto &msg : errors) {
			oss << "  - " << msg << "\n"; // collect errors
		}
		throw std::runtime_error(oss.str());
	}

	// shrink sample vector to actual number of loaded samples
	samples.samples.resize(samples.status.dbSize);
	std::cout << "Loaded " << samples.status.dbSize << " samples." << std::endl;
}

TrainSample DataBase::getSample(const size_t i) {
	TrainSample newSample;
	newSample.weight = samples.samples[i].weight;
	newSample.index = samples.samples[i].index;

	newSample.out = samples.samples[i].out;

	// copy input shape and strides
	newSample.input.shape = samples.samples[i].input.shape;
	newSample.input.strides = samples.samples[i].input.strides;

	// copy GPU or CPU data depending on global state
	if (nn::global::Tensor::getGpuState()) {
		newSample.input.gpu_data = samples.samples[i].input.gpu_data;
	} else {
		newSample.input.cpu_data = samples.samples[i].input.cpu_data;
	}
	return newSample;
}

} // namespace nn::model
