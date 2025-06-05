#ifndef AIMODEL
#define AIMODEL

#include "../src/model/config.hpp"
#include "../src/model/model.hpp"
#include <string>

struct prediction {
	const int index;
	const Global::ValueType value;
};

class AiModel {
  private:
	std::unique_ptr<model> _model;
	Config config;
	friend class BackPropagation;
	friend class Trainer;

  public:
	AiModel(const std::string &config_file, const bool use_visual = false);
	void run_model(const std::vector<Global::ValueType> &input);
	prediction getPrediction();
	Config &getConfig() { return config; }
	~AiModel() = default;
};

#endif // AIMODEL
