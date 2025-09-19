#ifndef MODEL
#define MODEL

#include "../src/model/optimizers.hpp"
#include "../src/visualizer/VisualizerController.hpp"
#include "dataBase.hpp"
#include <network/INetwork.hpp>

namespace nn::model {

constexpr int SECONDS_IN_MINUTE = 60;

const std::string TRAINING_HEADER = "Training Model";
const std::string EVALUATING_HEADER = "Evaluating Model";

const std::string SAVING_DATA_HEADER = "Saving parameters from: ";
const std::string LOADING_DATA_HEADER = "Loading parameters from: ";

struct modelResult {
	float dbSize;
	float currectPreSize;
	float percentage;
};

class Model {
  private:
	const Config config;

	std::vector<std::unique_ptr<INetwork>> network;
	visualizer::VisualManager visual;

	global::ValueType learningRate;
	std::unique_ptr<IOptimizer> optimizer;

	size_t batchCounter{0};

	void Forword(const global::Tensor &input, const int modelIndex);
	void Backward(global::Tensor &output, const global::ValueType weight);
	void updateWeights(const int batch_size);
	void resetNetworkGradient();
	global::ValueType getLoss(const global::Prediction &pre);

	global::ValueType runBackPropagation(
	    const Batch &batch, DataBase &db, const bool updateWeights);

	void printTrainingResult(
	    const std::chrono::high_resolution_clock::time_point &start,
	    const double error);

	void initModel();
	void initVisual();
	void initOptimizer();

	bool shouldRenderNet() const;

	float calculatePercentage(float currentSize, float totalSize);

	void trainModel(DataBase &trainedDataBase, DataBase &evaluateDataBase);

	size_t outputSize() const;
	const global::Tensor &getOutput() const;

	void setTraining(const bool state);
	void setTraining();
	void setNormal();
	void setEvaluating();

	bool autoEvaluating(const int i, DataBase &evaluateDataBase);
	void autoSave(const int i);

	void addFNN(const std::uint32_t width, ISubNetworkConfig &_config);
	void addCNN(const std::uint32_t width, ISubNetworkConfig &_config);

	std::uint32_t calculateSubNetWidth() const;

	void generateBatches(DataBase &db, std::vector<Batch> &batches);
	Batch &getBatch(DataBase &db, size_t &index, std::vector<Batch> &batches);

  public:
	Model(const std::string &config_filepath);
	~Model() = default;

	void runModel(const global::Tensor &input);
	void train(DataBase &dbT, DataBase &dbE);
	modelResult evaluateModel(
	    DataBase &dataBase, const bool cancleOnError = false,
	    const bool showProgressbar = true);

	void save(const std::string &file, const bool print = true);
	void load(const std::string &file, const bool print = true);

	global::Prediction getPrediction() const;
	std::vector<global::ValueType> getOut() const;

	void resetTraining();
};

} // namespace nn::model

#endif // MODEL
