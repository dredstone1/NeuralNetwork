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
	AiModel *model;
	BackPropagation backPropagation;
	void print_progress_bar(int current, int total);
	int last_progress;

  public:
	Trainer(AiModel *_model);
	void train();
	~Trainer() = default;
};

#endif // TRAINER_HPP
