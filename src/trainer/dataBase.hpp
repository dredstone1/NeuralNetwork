#ifndef DATABASE
#define DATABASE

#include "AiModel.hpp"
#include <random>
#include <string>
#include <vector>

typedef struct TrainSample {
	prediction _prediction;
	std::vector<double> input;
	TrainSample(prediction _pre, const int sampleInputSize) : _prediction(_pre), input(sampleInputSize, 0) {}
	TrainSample() : _prediction({0, 0}), input(0) {}
	~TrainSample() = default;
} TrainSample;

typedef struct Samples {
	const int sInputSize;
	std::vector<TrainSample> samples;
	size_t size() const { return samples.size(); }
	void add(TrainSample sample) { samples.push_back(sample); }
	Samples(const int sampleInputSize, const int _size) : sInputSize(sampleInputSize) {
		if (_size > 0) {
			samples.reserve(_size);
		}
	}
	~Samples() = default;
} Samples;

typedef struct Batch {
	std::vector<TrainSample *> samples;
	size_t size() const { return samples.size(); }
	Batch(const int length) {
		if (length > 0) {
			samples.resize(length, nullptr);
		}
	}
	~Batch() = default;
} Batch;

class DataBase {
  private:
	void getDataBaseStatus(const std::string &line);
	TrainSample read_line(const std::string &line);
	Samples *samples;
	int load();
	std::vector<Batch> batches;
	void generete_batches();

	TrainingConfig &config;
	size_t currentBatch;

	std::vector<int> shuffled_indices;
	std::mt19937 rng;

  public:
	DataBase(TrainingConfig &config_);
	size_t DataBaseLength() const { return samples ? samples->size() : 0; }
	Batch &get_Batch();
	~DataBase();
};

#endif // DATABASE
