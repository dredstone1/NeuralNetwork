#ifndef AIMODEL_HPP
#define AIMODEL_HPP

#include "../src/model/config.hpp"
#include "../src/model/model.hpp"
#include <string>

typedef struct prediction {
	const int index;
	const double value;
} prediction;

class AiModel {
  private:
	model *_model;
	Config *config;
	friend class BackPropagation;

  public:
	AiModel(std::string config_file, const bool use_visual);
	void run_model(const std::vector<double> &input);
	prediction getPrediction();
	Config &getConfig() { return *config; }
	~AiModel();
};

#endif // AIMODEL_HPP
