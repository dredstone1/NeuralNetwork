#ifndef DATABASE
#define DATABASE

#include "config.hpp"
#include <Globals.hpp>
#include <memory>
#include <random>

namespace nn::model {
struct TrainSample {
	global::Prediction prediction;
	global::ParamMetrix input;

	TrainSample(global::Prediction _pre, const int sampleInputSize)
	    : prediction(_pre),
	      input(sampleInputSize, 0) {}
	TrainSample()
	    : prediction(0, 0),
	      input(0) {}
};

struct Samples {
	const int sInputSize;
	std::vector<TrainSample> samples;

	Samples(const int sampleInputSize, const int _size)
	    : sInputSize(sampleInputSize) {
		samples.reserve(_size);
	}
	~Samples() = default;

	size_t size() const { return samples.size(); }
	void add(TrainSample sample) { samples.push_back(sample); }
};

struct Batch {
	std::vector<TrainSample *> samples;

	Batch(const int length) {
		samples.resize(length, nullptr);
	}
	~Batch() = default;

	size_t size() const { return samples.size(); }
};

class DataBase {
  private:
	std::unique_ptr<Samples> samples;
	std::vector<Batch> batches;
	size_t currentBatch;
	std::vector<int> shuffled_indices;
	std::mt19937 rng;

	const TrainingConfig &config;

	void getDataBaseStatus(const std::string &line);
	TrainSample readLine(const std::string &line);
	int load();
	void generateBatches();

  public:
	DataBase(const TrainingConfig &config);
	~DataBase() = default;

	size_t DataBaseLength() const { return samples ? samples->size() : 0; }
	Batch &getBatch();
};
} // namespace nn::model

#endif // DATABASE
