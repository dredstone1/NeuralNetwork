#ifndef MODEL
#define MODEL

#include "../src/model/dataBase.hpp"
#include "../src/visualizer/VisualizerController.hpp"
#include <memory>
#include <network/INetwork.hpp>
#include "../src/model/optimizers.hpp"

namespace nn::visualizer {
constexpr int BAR_WIDTH = 100;

class ProgressBar {
	const int total;
	const std::string header;

	int current{0};
	int last_percentage{-1};

  public:
	ProgressBar(const int total_, const std::string header_)
	    : total(total_),
	      header(header_ + ": ") {}
	~ProgressBar() = default;

	void printBar();

	ProgressBar operator++(int);
};
} // namespace nn::visualizer

namespace nn::model {
constexpr int SECONDS_IN_MINUTE = 60;

const std::string TRAINING_HEADER = "Training";
const std::string EVALUATING_HEADER = "Evaluating";

struct modelResult {
	int dbSize;
	int currectPreSize;
	float percentage;
};

class Model {
  private:
	const Config config;

	std::vector<std::unique_ptr<INetwork>> network;
	visualizer::VisualManager visual;

	global::ValueType learningRate;

	std::shared_ptr<IOptimizer> optimizer;

	void Forword(const global::ParamMetrix &input, const int modelIndex);

	void Backward(const global::ParamMetrix &output);
	void updateWeights(const int batch_size);

	void resetNetworkGradient();
	global::ValueType getLoss(const global::Prediction &pre);

	global::ValueType runBackPropagation(const Batch &batch, const bool updateWeights, global::Transformation transformation = nullptr);

	void printTrainingResult(
	    const std::chrono::high_resolution_clock::time_point &start,
	    const double error);

	void initModel();
	void initVisual();
	void initOptimizer();

	float calculatePercentage(size_t currentSize, size_t totalSize);

	modelResult evaluateModel(
	    DataBase &dataBase,
	    const bool cancleOnError = false,
	    const bool showProgressbar = true, global::Transformation transformation = nullptr);

	int outputSize();
	int inputSize();
	const global::ParamMetrix &getOutput() const;
    
    void setTraining(const bool state);

  public:
	Model(const std::string &config_filepath);
	~Model() = default;

	void runModel(const global::ParamMetrix &input);
	void train(const std::string &db_filename, global::Transformation transformation = nullptr);
	modelResult evaluateModel(
	    const std::string &db_filename,
	    const bool cancleOnError = false, global::Transformation transformation = nullptr);

	void save(const std::string &file);
	void load(const std::string &file);

	global::Prediction getPrediction();
};
} // namespace nn::model

#endif // MODEL
