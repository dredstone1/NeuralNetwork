#ifndef DATABASE
#define DATABASE

#include "../src/model/tensor_gpu.hpp"
#include <Globals.hpp>

namespace nn::model {
const std::string DATABASE_FILE_EXETENTION = ".nndb";

const std::string LOADING_DB_MESSAGE = "Loading DataBase: ";
const std::string FILE_NOT_FOUND_MESSAGE = "File not found: ";

struct TrainSample {
	global::Prediction pre;
	global::Tensor input;
	global::ValueType weight{1};

	TrainSample(const size_t sampleOutputSize, const size_t sampleInputSize)
	    : pre(sampleOutputSize, 0), input({sampleInputSize}, 0) {}
	TrainSample() : pre(0, 0), input({0}) {}
};

struct databaseStatus {
	size_t dbSize;
	size_t sampleInputSize;
	size_t sampleOutputSize;
};

struct Samples {
	databaseStatus status;
	std::vector<TrainSample> samples;
};

struct Batch {
	std::vector<size_t> samples;
	Batch(const size_t length) { samples.resize(length); }
	~Batch() = default;
	size_t size() const { return samples.size(); }
};

class Model;

class DataBase {
  private:
	std::vector<global::ValueType> tempData;

	databaseStatus getDataBaseStatus(const std::string &line);
	std::vector<size_t> getDataBaseInputShape(const std::string &line);
	int readLine(const std::string &line, TrainSample &sample);
	int loadData(const std::string &db_filename);

	friend Model;

  protected:
	Samples samples;

  public:
	DataBase() {}
	~DataBase() = default;

	void load(const std::vector<std::string> &fileNames);
	virtual TrainSample getSample(const size_t i);

	size_t DataBaseLength() const { return samples.status.dbSize; }
};
} // namespace nn::model

#endif // DATABASE
