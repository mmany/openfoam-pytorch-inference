#include <string>
#include <iostream>
#include <fstream>

#ifndef LIBTORCH_DEPENDENCY
#define LIBTORCH_DEPENDENCY
#include <torch/torch.h>
#include <torch/script.h>
#endif

#ifndef NLOHMANN_JSON_DEPENDENCY
#define NLOHMANN_JSON_DEPENDENCY
#include <nlohmann/json.hpp>
#endif


using json=nlohmann::json;

class Custom_normalizer {
	std::string load_filepath;	
	double mean[4] = {0,0,0,0};
	double std[4] = {0,0,0,0};
	json json_object;
	bool initialized;
	public:
		Custom_normalizer();
		Custom_normalizer(std::string json_filepath, bool deserialize_bool = true);
		torch::Tensor transform(torch::Tensor input);
		torch::Tensor inverse_transform(torch::Tensor input);
	//private:
		void serialize(std::string json_target_filepath);
		void deserialize(std::string json_filepath, bool updateMeanStd = false);

};
