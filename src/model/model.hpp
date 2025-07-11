#ifndef MODEL
#define MODEL

#include "../visualizer/VisualizerController.hpp"
#include "Globals.hpp"
#include "INetwork.hpp"
#include "dataBase.hpp"

namespace nn::model {
constexpr int BAR_WIDTH = 100;
constexpr int SECONDS_IN_MINUTE = 60;

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

	int lastProgress{-1};

	void Forword(const global::ParamMetrix &input, const int modelIndex);

	void Backward(const global::ParamMetrix &output);
	void update_weights(const int batch_size);

	bool isPredictionEqual(const global::ParamMetrix &pre1, const global::ParamMetrix &pre2);

	void resetNetworkGradient();
	global::ValueType getLoss(const global::Predictions &pre);

	global::ValueType run_back_propagation(const Batch &batch, const bool updateWeights);

	void print_progress_bar(const int current, const int total);
	void printTrainingResult(
	    const std::chrono::high_resolution_clock::time_point &start,
	    double error);

	void initModel();
	void initVisual();

  public:
	Model(const std::string &config_filepath);
	~Model() = default;

	void runModel(const global::ParamMetrix &input);
	void train(const std::string &db_filename);
	modelResult evaluateModel(const std::string &db_filename);

	void updateWeights(const global::ValueType learningRate);

	int outputSize();
	int inputSize();

	void save(const std::string &file);
	void load(const std::string &file);

	const global::ParamMetrix &getOutput() const;
};
} // namespace nn::model

#endif // MODEL
