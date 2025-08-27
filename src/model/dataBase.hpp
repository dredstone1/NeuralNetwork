#ifndef DATABASE
#define DATABASE

#include "config.hpp"
#include "tensor_gpu.hpp"
#include <Globals.hpp>
#include <random>
#include <vector>

namespace nn::model {
const std::string DATABASE_FILE_EXETENTION = ".nndb";

const std::string LOADING_DB_MESSAGE = "Loading DataBase: ";
const std::string FILE_NOT_FOUND_MESSAGE = "File not found: ";

struct TrainSample {
	global::Prediction pre;
	global::Tensor input;

	TrainSample(const size_t sampleOutputSize, const size_t sampleInputSize)
	    : pre(sampleOutputSize, 0), input({sampleInputSize}, 0) {}
	TrainSample() : pre(0, 0), input({0}) {}
};

struct databaseStatus {
	size_t dataBaseSize;
	size_t sampleInputSize;
	size_t sampleOutputSize;
};

struct Samples {
	databaseStatus status;
	std::vector<TrainSample> samples;

	Samples() {}
	~Samples() = default;

	size_t size() const { return status.dataBaseSize; }
	void add(TrainSample sample) { samples.push_back(sample); }
};

struct Batch {
	std::vector<TrainSample *> samples;
	Batch(const int length) { samples.resize(length, nullptr); }
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

	std::vector<global::ValueType> tempData;

	const TrainingConfig &config;

	databaseStatus getDataBaseStatus(const std::string &line);
	std::vector<size_t> getDataBaseInputShape(const std::string &line);
	int readLine(const std::string &line, TrainSample &sample);
	void generateBatches();

	int readLineFast(const char *ptr, const char *end, TrainSample &sample);
	int loadData(const std::string &db_filename);

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
