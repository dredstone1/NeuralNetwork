#ifndef MODEL
#define MODEL

#include "../src/model/optimizers.hpp"
#include "../src/visualizer/VisualizerController.hpp"
#include "tensor.hpp"
#include <dataBase.hpp>
#include <network/INetwork.hpp>

namespace nn::model {

/**
 * @brief Constant defining the number of seconds in one minute.
 */
constexpr int SECONDS_IN_MINUTE = 60;

/**
 * @brief Console headers used for progress reporting.
 */
const std::string TRAINING_HEADER = "Training Model";
const std::string EVALUATING_HEADER = "Evaluating Model";
const std::string SAVING_DATA_HEADER = "Saving parameters from: ";
const std::string LOADING_DATA_HEADER = "Loading parameters from: ";

struct Batch {
	std::vector<size_t> samples;
	Batch(const size_t length) { samples.resize(length); }
	~Batch() = default;
	size_t size() const { return samples.size(); }
};

/**
 * @struct modelResult
 * @brief Holds evaluation results for the model.
 */
struct modelResult {
	float dbSize;
	float currectPreSize;
	float percentage;
};

/**
 * @class Model
 * @brief Core class that manages training, evaluation, saving/loading, and running
 *        deep learning models composed of sub-networks (CNN/FNN...).
 */
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
	global::ValueType getLoss(const size_t index, const global::Tensor &out);

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
	/**
	 * @brief Construct a new Model from a configuration file.
	 * @param config_filepath Path to the configuration file.
	 */
	Model(const std::string &config_filepath);

	~Model() = default;

	/**
	 * @brief Run forward pass of the model with given input.
	 * @param input Input tensor to feed into the model.
	 */
	void runModel(const global::Tensor &input);

	/**
	 * @brief Train the model on given training and evaluation datasets.
	 * @param dbT Training database.
	 * @param dbE Evaluation database.
	 */
	void train(DataBase &dbT, DataBase &dbE);

	/**
	 * @brief Evaluate model accuracy on given dataset.
	 * @param dataBase Dataset for evaluation.
	 * @param cancleOnError Whether to stop on error.
	 * @param showProgressbar Whether to display progress bar.
	 * @return modelResult Structure with evaluation statistics.
	 */
	modelResult evaluateModel(
	    DataBase &dataBase, const bool cancleOnError = false,
	    const bool showProgressbar = true);

	/**
	 * @brief Save model parameters to file.
	 * @param file Output file path.
	 * @param print Whether to print progress.
	 */
	void save(const std::string &file, const bool print = true);

	/**
	 * @brief Load model parameters from file.
	 * @param file Input file path.
	 * @param print Whether to print progress.
	 */
	void load(const std::string &file, const bool print = true);

	/**
	 * @brief Get final prediction of the model after forward pass.
	 * @return global::Prediction Prediction result.
	 */
	global::Prediction getPrediction() const;

	/**
	 * @brief Get raw output values of the model.
	 * @return Vector of output values.
	 */
	std::vector<global::ValueType> getOut() const;

	/**
	 * @brief Reset training state without changing model weights.
	 */
	void resetTraining();
};

} // namespace nn::model

#endif // MODEL
