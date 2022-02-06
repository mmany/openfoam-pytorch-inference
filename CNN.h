//#include <ATen/core/ivalue.h>
//#include <ATen/core/TensorBody.h>
//#include <string>
//#include <torch/torch.h>
//#include <torch/script.h>
#include "Custom_normalizer.h"
#include "volFieldsFwd.H"
#include <vector>

#ifndef OPENFOAM_DEPENDENCY
#define OPENFOAM_DEPENDENCY
#include "fvCFD.H"
//#include "fvMesh.H"
//#include "volFieldsFwd.H"
//#include "UList.H"
#endif


//# define M_PI           3.14159265358979323846
// Already defined in math.h


class CNN {

	torch::jit::script::Module module;
	Custom_normalizer input_normalizer;
	Custom_normalizer output_normalizer;

	// For under_relaxation
	torch::Tensor old_nut = torch::zeros(1);
	torch::Tensor current_nut = torch::zeros(0);


	public:



		// Constructors 
		CNN(std::string model_path);
		CNN(std::string model_path, std::string input_normalizer_path, std::string output_normalizer_path);


		// Member functions
		torch::Tensor predict(torch::Tensor input, float alpha=1); // alpha is the under-relaxation factor (1 == no UR ; 0 == no prediction)
		torch::Tensor predict_test(torch::Tensor input); // For in-between shape printing
		void save_tensor(torch::Tensor tensor_to_be_saved, std::string filename);


		//For OF field conversion to torch::Tensor
		torch::Tensor convertToTensor_test(Foam::volVectorField& U0, Foam::volVectorField& U1);
		torch::Tensor convertToTensor(Foam::volVectorField& U0, Foam::volVectorField& U1);
		void updateFoamFieldChannelFlow(torch::Tensor& t, Foam::volScalarField& nut0, Foam::volScalarField& nut1);
		void updateFoamFieldChannelFlow_velocity(torch::Tensor& t, Foam::volVectorField& U0, Foam::volVectorField& U1);
		Foam::volScalarField convertToFoamField(const torch::Tensor &t, const std::string& field_name, const Foam::Time& runTime, const Foam::fvMesh& mesh, int uqNode);


		// Output transforms (smoothing etc...)
		torch::Tensor gaussian_smoothing(torch::Tensor input, int kernel_size, float sigma);
		torch::Tensor under_relaxation(float alpha);
		torch::Tensor under_relaxation(torch::Tensor old, torch::Tensor current, float alpha);


		// Resize tensor functions
		torch::Tensor resizeToNetworkSize(const torch::Tensor& input);
		torch::Tensor resizeToOriginal(const torch::Tensor& input,std::vector<int64_t> size);


		//Utility member functions
		float get_max_rate_of_change();
		torch::Tensor set_wall_to_zero(const torch::Tensor& input);
		void set_inlet_nut(Foam::volScalarField& field, const float value);
		void set_inlet_nut(Foam::volVectorField& field, const float value);
		//void plot_tensor(torch::Tensor to_be_plotted);
		
		 
		// Printing Foam Fields
		void printFoamField(const Foam::volVectorField& vectorField);
		void printFoamField(const Foam::volScalarField& scalarField);
		void printFoamFieldNodes(const Foam::volVectorField& vectorField);
		void printFoamFieldNodes(const Foam::volScalarField& scalarField);
 		void printInletNodes(const Foam::volScalarField& scalarField); // For testing
		void printInletNodesBis(const Foam::volScalarField& scalarField);

		// Loading tensor in a file
		torch::Tensor loadTensorFromContainer(const std::string container_filepath, const std::string key);
};
