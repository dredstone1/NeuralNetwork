#ifndef DATABASE
#define DATABASE

#include "config.hpp"
#include <random>
#include <vector>

namespace nn::model {
const std::string DATABASE_FILE_EXETENTION = ".nndb";

struct TrainSample {
	global::Prediction pre;
	global::Tensor input;

	TrainSample(const size_t sampleOutputSize, const size_t sampleInputSize)
	    : pre(sampleOutputSize, 0),
	      input({sampleInputSize}, 0) {}
	TrainSample()
	    : pre(0, 0),
	      input({0}) {}
};

struct Samples {
	size_t sInputSize;
	size_t sOutputSize;

	std::vector<TrainSample> samples;

	Samples() {}
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
	Samples samples;
	std::vector<Batch> batches;
	size_t currentBatch;
	std::vector<int> shuffled_indices;
	std::mt19937 rng;

	const TrainingConfig &config;

	void getDataBaseStatus(const std::string &line);
	TrainSample readLine(const std::string &line);
	void generateBatches();

  public:
	DataBase(const TrainingConfig &config);
	~DataBase() = default;

	int load(const std::string &db_filename);
	int load(const std::vector<std::string> &db_filenames);
	TrainSample &getSample(const int i) { return samples.samples[i]; }

	size_t DataBaseLength() const { return samples.size(); }
	Batch &getBatch();
};
} // namespace nn::model

#endif // DATABASE
