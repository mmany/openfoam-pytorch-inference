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

/** @brief Class used to provide the tools to normalize and denormalize a given tensor, using the normalizers generated during the training of a network.

    
    @author Maheindrane Many
    @date March 2022
    */
class CustomNormalizer {
	std::string load_filepath;/**< attribute used to store filepath of normalizer */	
	//double mean[4] = {0,0,0,0}; /**< single dimension array used to store the mean values for the normalization process*/
	//double std[4] = {0,0,0,0};/**< single dimension array used to store the standard deviation values for the normalization process*/
	json json_object; /**< atribute used for the loading of the json object*/
	bool initialized; /**< boolean used to check if the object is initialized with the mean and std values*/
	public:
	/**
	 * Default constructor, only creates the object, not recommended
	 */
		CustomNormalizer();
	/**
	 * Recommended constructor. Overload of the default constructor which directly initializes the means and stds with the specified normalizer path.
	 @param json_filepath path to normalize, stored as a JSON object.
	 @param deserialize_bool if true, updates the mean and std values
	 */
		CustomNormalizer(std::string json_filepath, bool deserialize_bool = true);
	/**
	 * Function to perform normalization of every individiual channel of the tensor with the given mean and std. The performed operation is \f$ output[i] = \frac{input[i]-mean[i]}{std[i]}\f$. Returns the normalized tensor.
	 @param input torch tensor to be normalized
	 */
		torch::Tensor transform(torch::Tensor input);
	/**
	 * Function to perform denormalization of every individiual channel of the tensor with a given mean and std. The performed operation is \f$ output[i] = input[i]*std[i] + mean[i]\f$. Returns the denormalized tensor.
	 @param input torch tensor to be denormalized
	 */
		torch::Tensor inverse_transform(torch::Tensor input);
	//private:
	/**
	 * saves the mean and std vectors in the specified filepath as a JSON dictionnary
	 @param json_target_filepath absolute path of the json object to be saved
	 */
		void serialize(std::string json_target_filepath);
	/**
	 * Loads the mean and std deviation vectors from a JSON file and updates the corresponding class attributes.
	 @param json_filepath absolute path of the normalizer to be loaded
	 */
		void deserialize(std::string json_filepath);

};
