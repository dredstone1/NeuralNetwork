#ifndef DATABASE
#define DATABASE

#include <AiModel.hpp>
#include <random>

namespace nn::training {
struct TrainSample {
	Prediction prediction;
	global::ParamMetrix input;

	TrainSample(Prediction _pre, const int sampleInputSize)
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
	model::TrainingConfig &config;
	size_t currentBatch;
	std::vector<int> shuffled_indices;
	std::mt19937 rng;

	void getDataBaseStatus(const std::string &line);
	TrainSample read_line(const std::string &line);
	int load();
	void generete_batches();

  public:
	DataBase(model::TrainingConfig &config_);
	~DataBase() = default;

	size_t DataBaseLength() const { return samples ? samples->size() : 0; }
	Batch &get_Batch();
};
} // namespace nn::training

#endif // DATABASE
