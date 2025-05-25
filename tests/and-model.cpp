#include <AiModel.hpp>
#include <iostream>
#include <trainer.hpp>

using namespace std;

enum mode {
	load = 0,   // 0001
	save = 1,   // 0010
	train = 2,  // 0100
	visual = 4, // 1000
};

int main(int argc, char *argv[]) {
	if (argc > 1) {
		int mods = 0;
		char *arg = argv[1];
		while (*(arg) != '\0') {
			switch (*arg) {
			case 'l':
				mods |= load;
				break;
			case 's':
				mods |= save;
				break;
			case 't':
				mods |= train;
				break;
			case 'v':
				mods |= visual;
				break;
			}

			arg++;
		}

		AiModel model("config.json", mods & visual);

		// if (mods & load) {
		// 	model.load("model1", mods & visual);
		// }

		if (mods & train) {
			int batch_size = 2, batch_count = 10;

			double learning_rate = 0.001;

			Trainer trainer(&model);

			trainer.train();
		}

		// if (mods & save) {
		// 	model.save("model1");
		// }

		int num = 0;
		string str_num;
		while (num != 5) {
			cout << "Enter an integer: ";
			getline(cin, str_num);
			num = stoi(str_num);
			if (num == 5)
				break;
			vector<double> input(2, 0);
			input[num] = 1;
			model.run_model(input);
			printf("prediction: %d, %f\n", model.getPrediction().index, model.getPrediction().value);
		}

		return 0;
	}

	cout << "Need at least 1 input!" << endl;
	return 1;
}
