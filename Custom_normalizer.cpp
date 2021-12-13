#include "Custom_normalizer.h"
//#include <ATen/Functions.h>
//#include <torch/serialize/input-archive.h>
//#include <typeinfo>

using json = nlohmann::json;

Custom_normalizer::Custom_normalizer(){
	//this->mean = 0;
	//this->std = 0;
}

Custom_normalizer::Custom_normalizer(std::string json_filepath, bool deserialize_bool){
	//this->mean = 0;
	//this->std = 0;
	this->load_filepath = json_filepath;

	if(deserialize_bool){
		//Read the given json file 
		this->deserialize(json_filepath);
		std::cout << "Initialized Custom_Normalizer with file : "<< json_filepath << std::endl;
		this->initialized = true;
	} else {
		std::cout << "Initialized Custom_Normalizer with file, not deserialized : "<< json_filepath << std::endl;
	}
}

void Custom_normalizer::serialize(std::string json_target_filepath){
	std::cout << "Serializing Custom normalizer : " << json_target_filepath << std::endl;
	json j;
	j["mean"] = this->mean;
	j["std"] = this->std;
	std::ofstream f(json_target_filepath);
	f << j.dump(4);
	f.close();

}

void Custom_normalizer::deserialize(std::string json_filepath, bool updateMeanStd){
	std::cout << "deserialize : " << json_filepath << std::endl;
	std::ifstream json_filestream(json_filepath/*, std::ifstream::binary*/);
	json_filestream >> json_object;
	json_filestream.close();
	std::cout << "Successfully deserialized." << std::endl;
	std::cout << "Found Keys : " << std::endl;
	for (auto it: this->json_object.items()){
		std::cout << it.key() << " , " << it.value() << std::endl;
		if(updateMeanStd){
			if(it.key() == "mean"){
				this->mean[3] = it.value();
			} else if(it.key() == "std"){
				this->std[3] = it.value();
			}
		}
		//std::cout << "Type of it.value() : "  << typeid(it.value()).name() << std::endl;
		//does not return the expected output
	}
}

torch::Tensor Custom_normalizer::transform(torch::Tensor input){
	if(!this->initialized){
		std::cout << "Unable to perform transform, normalizer not initialized, returning input" << std::endl;
		return input;
	} else {
		torch::Tensor output = torch::ones(input.sizes());
		for(int k=0;k<input.size(1);k++){
			output.index({0,k,"..."}) = input.index({0,k,"..."}).subtract(float(this->json_object["mean"][k])).divide(float(this->json_object["std"][k]));
			//What it roughly does in syntaxically incorrect code
			//output[0][k][:][:] = (input[0][k][:][:]- this->json_object["mean"][k])/this->json_object["std"][k];
		}
		return output;
	}
}

torch::Tensor Custom_normalizer::inverse_transform(torch::Tensor input){
	if(!this->initialized){
		std::cout << "Unable to perform inverse transform, normalizer not initialized, returning inptu" << std::endl;
		return input;
	} else {
		torch::Tensor out = torch::ones(input.sizes());
		for(int k=0;k<input.size(1);k++){
			//std::cout << k << std::endl;
			//std::cout << input.size(1) << std::endl;
			out.index({0,k,"..."}) = input.index({0,k,"..."}).multiply(float(this->json_object["std"][k])).add(float(this->json_object["mean"][k]));
			// What it roughly does.
			//output[:,k,:,:] = (input[:,k,:,:] * this->std[k] + this->mean[k])/;
		}
		return out;
	}
}
