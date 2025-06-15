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
	~TrainSample() = default;
};

struct Samples {
	const int sInputSize;
	std::vector<TrainSample> samples;

	size_t size() const { return samples.size(); }
	void add(TrainSample sample) { samples.push_back(sample); }
	Samples(const int sampleInputSize, const int _size)
	    : sInputSize(sampleInputSize) {
		if (_size > 0) {
			samples.reserve(_size);
		}
	}
	~Samples() = default;
};

struct Batch {
	std::vector<TrainSample *> samples;

	size_t size() const { return samples.size(); }
	Batch(const int length) {
		if (length > 0) {
			samples.resize(length, nullptr);
		}
	}
	~Batch() = default;
};

class DataBase {
  private:
	void getDataBaseStatus(const std::string &line);
	TrainSample read_line(const std::string &line);
	std::unique_ptr<Samples> samples;
	int load();
	std::vector<Batch> batches;
	void generete_batches();

	model::TrainingConfig &config;
	size_t currentBatch;

	std::vector<int> shuffled_indices;
	std::mt19937 rng;

  public:
	DataBase(model::TrainingConfig &config_);
	size_t DataBaseLength() const { return samples ? samples->size() : 0; }
	Batch &get_Batch();
	~DataBase() = default;
};
} // namespace nn::training

#endif // DATABASE
