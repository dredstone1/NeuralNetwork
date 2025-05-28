#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "../src/trainer/backPropagation.hpp"
#include "../src/trainer/dataBase.hpp"
#include "AiModel.hpp"

#define GRAPH_RESOLUTION 100
#define BAR_WIDTH 100
#define SECONDS_IN_MINUTE 60

class Trainer {
  private:
	TrainingConfig &config;
	DataBase dataBase;
	AiModel &model;
	BackPropagation backPropagation;
	void print_progress_bar(const int current, const int total);
	int last_progress;

  public:
	Trainer(AiModel &_model) : config(_model.getConfig().config_data.training_config), dataBase(config), model(_model), backPropagation(_model), last_progress(-1) {}
	void train();
	~Trainer() = default;
};

#endif // TRAINER
