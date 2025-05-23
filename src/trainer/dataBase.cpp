#include "dataBase.hpp"
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string.h>
#include <string>
#include <vector>

using namespace std;

DataBase::DataBase(const string &_file_name) : file_name(_file_name) {
	load();
}

TrainSample DataBase::read_line(const string &line) {
	char cstr[100];
	strcpy(cstr, line.c_str());
	char *token = strtok(cstr, " ");
	int best_next_move = stoi(token);
	TrainSample new_sample({best_next_move, 1.f}, samples->sInputSize);

	for (int i = 0; (i < samples->sInputSize && token != NULL); i++) {
		token = strtok(NULL, ":");
		new_sample.input[i] = stoi(token);
	}

	return new_sample;
}

void DataBase::getDataBaseStatus(const string &line) {
	char cstr[100];
	strcpy(cstr, line.c_str());
	char *token = strtok(cstr, " ");

	int dataBaseSize = stoi(token);
	int sampleSize = stoi(token = strtok(NULL, " "));
	samples = new Samples(sampleSize, dataBaseSize);
}

int DataBase::load() {
	ifstream file(file_name + ".txt");
	if (!file.is_open()) {
		cout << "File not found: " << file_name << endl;
		return 1;
	}
	string line;
	getline(file, line);
	getDataBaseStatus(line);

	while (samples->size() >= 0 && getline(file, line)) {
		samples->add(read_line(line));
	}

	samples->samples.shrink_to_fit();
	cout << "Loaded " << DataBaseLength() << " boards." << endl;
	file.close();

	return 0;
}

Batch DataBase::get_Batch(const int batch_size) {
	Batch new_batch(batch_size, samples->sInputSize);

	for (int i = 0; i < batch_size; i++) {
		new_batch.add((samples->samples.data() + i));
	}

	return new_batch;
}
