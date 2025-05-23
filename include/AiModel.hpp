#ifndef AIMODEL_HPP
#define AIMODEL_HPP

#include "../src/model/model.hpp"
#include <string>

#define MODEL_FILE_EXTENSION ".model"

typedef struct prediction {
    const int index;
    const double value;
} prediction;

class AiModel {
  private:
	model *_model;
	friend class BackPropagation;

  public:
	AiModel();
	AiModel(string file_name);

	int load(const string file_name);
	void save(const string file_name);
	void run_model(const vector<double> &input);
	prediction getPrediction();
	~AiModel();
};

#endif // AIMODEL_HPP
