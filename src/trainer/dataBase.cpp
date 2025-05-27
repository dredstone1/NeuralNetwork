#include "dataBase.hpp"
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <ostream>

using namespace std;

DataBase::DataBase(TrainingConfig &config_)
    : samples(nullptr), config(config_), currentBatch(0) {
	random_device rd;
	rng = mt19937(rd());

	load();
	shuffled_indices.resize(samples->size());
	iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
	generete_batches();
}

DataBase::~DataBase() {
	delete samples;
	samples = nullptr;
}

TrainSample DataBase::read_line(const string &line) {
	char cstr[100];
	strncpy(cstr, line.c_str(), sizeof(cstr) - 1);
	cstr[sizeof(cstr) - 1] = '\0';

	char *context = nullptr;
	char *token = strtok_r(cstr, " ", &context);
	int best_next_move = stoi(token);

	TrainSample new_sample({best_next_move, 1.f}, samples->sInputSize);

	for (int i = 0; i < samples->sInputSize; ++i) {
		token = strtok_r(NULL, " ", &context);
		new_sample.input[i] = stod(token);
	}
	return new_sample;
}

void DataBase::getDataBaseStatus(const string &line) {
	char cstr[100];
	strncpy(cstr, line.c_str(), sizeof(cstr) - 1);
	cstr[sizeof(cstr) - 1] = '\0';

	char *context = nullptr;
	char *token = strtok_r(cstr, " ", &context);
	int dataBaseSize = stoi(token);

	token = strtok_r(NULL, " ", &context);
	int sampleSize = stoi(token);

	if (samples) {
		delete samples;
	}
	samples = new Samples(sampleSize, dataBaseSize);
}

int DataBase::load() {
	ifstream file(config.db_filename + ".txt");
	if (!file.is_open()) {
		cout << "File not found: " << config.db_filename << endl;
		return 1;
	}

	string line;
	getline(file, line);
	getDataBaseStatus(line);

	while (getline(file, line)) {
		if (line.empty() || line.find_first_not_of(" \t\n\v\f\r") == string::npos) {
			continue;
		}
		samples->add(read_line(line));
	}

	if (samples->samples.capacity() > samples->size()) {
		samples->samples.shrink_to_fit();
	}

	cout << "Loaded " << (samples ? samples->size() : 0) << " samples." << endl;
	file.close();

	return 0;
}

void DataBase::generete_batches() {
	shuffle(shuffled_indices.begin(), shuffled_indices.end(), rng);

	batches.clear();
	size_t num_batches_expected = (samples->size() + config.batch_size - 1) / config.batch_size;
	batches.reserve(num_batches_expected);

	for (size_t i = 0; i < samples->size(); i += config.batch_size) {
		size_t current_batch_actual_size = min((size_t)config.batch_size, samples->size() - i);

		if (current_batch_actual_size == 0)
			break;

		batches.emplace_back(current_batch_actual_size);
		Batch &new_batch = batches.back();

		for (size_t j = 0; j < current_batch_actual_size; ++j) {
			int sample_original_index = shuffled_indices[i + j];
			new_batch.samples_ptrs[j] = &samples->samples[sample_original_index];
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
