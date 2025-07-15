#ifndef MODEL
#define MODEL

#include "../visualizer/VisualizerController.hpp"
#include "dataBase.hpp"
#include <network/INetwork.hpp>

namespace nn::model {
constexpr int BAR_WIDTH = 100;
constexpr int SECONDS_IN_MINUTE = 60;

const std::string TRAINING_HEADER = "Training";
const std::string EVALUATING_HEADER = "Evaluating";

struct modelResult {
	int dbSize;
	int currectPreSize;
	float percentage;
};

class ProgressBar {
	const int total;
	const std::string header;

	int current;
	int last_percentage{-1};

  public:
	ProgressBar(const int total_, const std::string header_)
	    : total(total_),
	      header(header_ + ": ") {}
	~ProgressBar() = default;

	void printBar();

	ProgressBar operator++(int);
};

class Model {
  private:
	const Config config;

	std::vector<std::unique_ptr<INetwork>> network;
	visualizer::VisualManager visual;

	global::ValueType learningRate;

	void Forword(const global::ParamMetrix &input, const int modelIndex);

	void Backward(const global::ParamMetrix &output);
	void updateWeights(const int batch_size);

	void resetNetworkGradient();
	global::ValueType getLoss(const global::Prediction &pre);

	global::ValueType runBackPropagation(const Batch &batch, const bool updateWeights, global::Transformation transformation = nullptr);

	void printTrainingResult(
	    const std::chrono::high_resolution_clock::time_point &start,
	    double error);

	void initModel();
	void initVisual();

	float calculatePercentage(size_t currentSize, size_t totalSize);

	modelResult evaluateModel(
	    DataBase &dataBase,
	    const bool cancleOnError = false,
	    const bool showProgressbar = true, global::Transformation transformation = nullptr);

  public:
	Model(const std::string &config_filepath);
	~Model() = default;

	void runModel(const global::ParamMetrix &input);
	void train(const std::string &db_filename, global::Transformation transformation = nullptr);
	modelResult evaluateModel(
	    const std::string &db_filename,
	    const bool cancleOnError = false, global::Transformation transformation = nullptr);

	int outputSize();
	int inputSize();

	void save(const std::string &file);
	void load(const std::string &file);

	const global::ParamMetrix &getOutput() const;
};
} // namespace nn::model

#endif // MODEL
