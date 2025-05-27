#ifndef DATABASE_HPP
#define DATABASE_HPP

#include "AiModel.hpp"
#include <random>
#include <string>
#include <vector>

using namespace std;

typedef struct TrainSample {
	prediction _prediction;
	vector<double> input;
	TrainSample(prediction _pre, const int sampleInputSize) : _prediction(_pre), input(sampleInputSize, 0) {}
	TrainSample() : _prediction({0, 0}), input(0) {}
} TrainSample;

typedef struct Samples {
	const int sInputSize;
	vector<TrainSample> samples;
	size_t size() const { return samples.size(); }
	void add(TrainSample sample) { samples.push_back(sample); }
	Samples(const int sampleInputSize, const int _size) : sInputSize(sampleInputSize) {
		if (_size > 0) {
			samples.reserve(_size);
		}
	}
} Samples;

typedef struct Batch {
	vector<TrainSample *> samples_ptrs;
	size_t size() const { return samples_ptrs.size(); }
	Batch(const int length) {
		if (length > 0) {
			samples_ptrs.resize(length, nullptr);
		}
	}
	Batch() = default;
} Batch;

class DataBase {
  private:
	void getDataBaseStatus(const string &line);
	TrainSample read_line(const string &line);
	Samples *samples;
	int load();
	vector<Batch> batches;
	void generete_batches();

	TrainingConfig &config;
	size_t currentBatch;

	vector<int> shuffled_indices;
	mt19937 rng;

  public:
	DataBase(TrainingConfig &config_);
	size_t DataBaseLength() const { return samples ? samples->size() : 0; }
	Batch &get_Batch();
	~DataBase();
};

#endif // DATABASE
