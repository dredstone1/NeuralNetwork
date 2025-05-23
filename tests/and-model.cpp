#include <AiModel.hpp>
#include <iostream>
#include <trainer.hpp>

using namespace std;

int main(int argc, char *argv[]) {
	if (argc > 1) {
		AiModel model;
		char *arg = argv[1];
		while (*(arg) != '\0') {
			switch (*arg) {
			case 'l':
				model.load("model1");
				break;
			case 's':
				model.save("model1");
				break;
			case 't':
				int batch_size = 64, batch_count = 10000;

				double learning_rate = 0.000001;

				Trainer trainer(
				    "database",
				    &model,
				    batch_size,
				    batch_count,
				    learning_rate);

				trainer.train();
				model.save("model1");
				break;
			}
			arg++;
		}

		return 0;
	}

	cout << "Need at least 1 input!" << endl;
	return 1;
}
