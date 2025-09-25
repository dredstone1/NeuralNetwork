#ifndef DATABASE
#define DATABASE

#include "../src/model/tensor_gpu.hpp"
#include "tensor.hpp"

namespace nn::model {
const std::string DATABASE_FILE_EXETENTION = ".nndb";

const std::string LOADING_DB_MESSAGE = "Loading DataBase: ";
const std::string FILE_NOT_FOUND_MESSAGE = "File not found: ";

/**
 * @brief Represents a single training sample in the dataset.
 *
 * A training sample consists of:
 * - The expected prediction (label/output).
 * - The input tensor (features).
 * - An optional weight (useful for weighted training).
 */
struct TrainSample {
	global::Tensor input;        ///< Input features for the sample.
	global::ValueType weight{1}; ///< Sample weight (default = 1).

	global::Tensor *out = nullptr;
	size_t index = 0;

	/**
	 * brief Construct a training sample with given input/output sizes.
	 * @param sampleOutputSize Number of expected output values.
	 * @param sampleInputSize  Number of input features.
	 */
	TrainSample(const size_t sampleInputSize)
	    : input({sampleInputSize}, 0) {
	}

	/**
	 * @brief Construct an empty training sample (zero sizes).
	 */
	TrainSample() : input({0}) {}
	~TrainSample() {
		if (out) {
			delete out;
		}
	}
};

enum class OutType {
	Classify,
	Statistic,
};

/**
 * @brief Stores general information about the dataset.
 */
struct databaseStatus {
	size_t dbSize{0};           ///< Total number of samples in the database.
	size_t sampleInputSize{0};  ///< Input dimension of each sample.
	size_t sampleOutputSize{0}; ///< Output dimension of each sample.

	OutType outputType{OutType::Classify};
};

/**
 * @brief Holds the actual dataset and its metadata.
 */
struct Samples {
	databaseStatus status;            ///< General dataset information.
	std::vector<TrainSample> samples; ///< All training samples.
};

class Model;

/**
 * @brief Manages dataset loading, parsing, and sample access.
 *
 * This class is intended to be inherited when you need custom dataset
 * handling. Override `getSample()` to implement special sampling logic
 * (e.g., data augmentation, synthetic samples, streaming).
 *
 * Responsibilities:
 * - Load dataset files from disk.
 * - Store training samples and metadata.
 * - Provide sample access for training and evaluation.
 */
class DataBase {
  private:
	std::vector<global::ValueType> tempDataI;
	std::vector<global::ValueType> tempDataO;

	/**
	 * @brief Extract dataset metadata (sizes) from a header line.
	 * @param line The header line from the dataset file.
	 * @return A populated databaseStatus structure.
	 */
	databaseStatus getDataBaseStatus(const std::string &line);

	/**
	 * @brief Parses a single line from a dataset file into a TrainSample
	 *
	 * This method parses a line from the dataset file and extracts the prediction
	 * index, weight, and input features. The expected format is:
	 * - 'p' followed by prediction index
	 * - 'w' followed by sample weight
	 * - Numeric values for input features
	 *
	 * @param line The line to parse from the dataset file
	 * @param sample Output parameter: filled training sample
	 *
	 * @throws std::runtime_error If the line format is invalid or incomplete
	 */
	void readLine(const std::string &line, TrainSample &sample);

	/**
	 * @brief Load all samples from a given dataset file.
	 * @param db_filename Path to the dataset file.
	 * @return 0 if successful, non-zero on error.
	 */
	int loadData(const std::string &db_filename);

	friend Model;

  protected:
	Samples samples;

  public:
	/**
	 * @brief Construct an empty database.
	 */
	DataBase() {}

	~DataBase() = default;

	/**
	 * @brief Load multiple dataset files into memory.
	 * @param fileNames List of file paths to load.
	 */
	void load(const std::vector<std::string> &fileNames);

	/**
	 * @brief Retrieve a training sample by index.
	 *
	 * Can be overridden by derived classes to provide custom
	 * sample retrieval strategies (e.g., shuffling, augmentation).
	 */
	virtual TrainSample getSample(const size_t i);

	size_t DataBaseLength() const { return samples.status.dbSize; }
};
} // namespace nn::model

#endif // DATABASE
