#ifndef MODEL
#define MODEL

#include "../src/model/dataBase.hpp"
#include "../src/model/optimizers.hpp"
#include "../src/visualizer/VisualizerController.hpp"
#include "tensor.hpp"
#include <memory>
#include <network/INetwork.hpp>
#include <vector>

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

inline nn::global::Transformation dt = [](const nn::global::Tensor &p) {
	return p;
};

class Model {
  private:
	const Config config;

	std::vector<std::unique_ptr<INetwork>> network;
	visualizer::VisualManager visual;

	global::ValueType learningRate;
	std::unique_ptr<IOptimizer> optimizer;

	void Forword(const global::Tensor &input, const int modelIndex);

	void Backward(const global::Tensor &output);
	void updateWeights(const int batch_size);

	void resetNetworkGradient();
	global::ValueType getLoss(const global::Prediction &pre);

	global::ValueType runBackPropagation(
	    const Batch &batch,
	    const bool updateWeights,
	    global::Transformation transformation = dt);

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
	    const bool showProgressbar = true,
	    global::Transformation transformation = dt);
	void trainModel(
	    DataBase &trainedDataBase,
	    DataBase &evaluateDataBase,
	    global::Transformation transformationB = dt,
	    global::Transformation transformationE = dt);

	size_t outputSize() const;
	size_t inputSize() const;
	const global::Tensor &getOutput() const;

	void setTraining();
	void setNormal();
	void setEvaluating();

	bool autoEvaluating(
	    const int i,
	    DataBase &evaluateDataBase,
	    global::Transformation transformationE);

	void autoSave(const int i);

    void addFNN(const std::uint32_t width, ISubNetworkConfig &_config);
    void addCNN(const std::uint32_t width, ISubNetworkConfig &_config);

    std::uint32_t calculateSubNetWidth() const;

  public:
	Model(const std::string &config_filepath);
	~Model() = default;

	void runModel(const global::Tensor &input);
	void train(
	    const std::string &db_filename,
	    global::Transformation transformationB = dt,
	    global::Transformation transformationE = dt);
	void train(
	    const std::vector<std::string> &db_filename,
	    global::Transformation transformationB = dt,
	    global::Transformation transformationE = dt);
	modelResult evaluateModel(
	    const std::string &db_filename,
	    const bool cancleOnError = false,
	    global::Transformation transformation = dt);

	void save(const std::string &file);
	void load(const std::string &file);

	global::Prediction getPrediction() const;
};
} // namespace nn::model

#endif // MODEL
