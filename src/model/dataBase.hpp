#ifndef DATABASE
#define DATABASE

#include "config.hpp"
#include <Globals.hpp>
#include <memory>
#include <random>

namespace nn::model {
const std::string DATABASE_FILE_EXETENTION = ".nndb";

struct TrainSample {
	global::Predictions output;
	global::ParamMetrix input;

	TrainSample(const int sampleOutputSize, const int sampleInputSize)
	    : output(sampleOutputSize, 0),
	      input(sampleInputSize, 0) {}
	TrainSample()
	    : output(0, 0),
	      input(0) {}
};

enum class SamplesMode {
	classification,
	full,
};

struct Samples {
	const int sInputSize;
	const int sOutputSize;

	const SamplesMode sMode;

	std::vector<TrainSample> samples;

	Samples(
	    const int sampleInputSize,
	    const int sampleOutputSize,
	    const SamplesMode sampleMode,
	    const int _size)
	    : sInputSize(sampleInputSize),
	      sOutputSize(sampleOutputSize),
	      sMode(sampleMode) {
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
