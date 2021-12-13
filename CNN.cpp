#include "CNN.h"
#include "ATen/Functions.h"
#include "torch/nn/modules/conv.h"
//#include "UList.H"
//#include <iostream>
//#include "volFieldsFwd.H"
//#include <ATen/core/ivalue.h>
//#include <torch/serialize.h>
//#include <vector
namespace F = torch::nn::functional;

CNN::CNN(std::string model_path){
	std::cout << "*************************************************" << std::endl;
	std::cout << "-------Initializing CNN with given model --------" << std::endl;
	std::cout << "*************************************************" << std::endl << std::endl;
	// ADD EXCEPTIOSN
	// Loading the CNN model
	std::cout << "Loading Torchscript model : "+model_path << std::endl;
	this->module = torch::jit::load(model_path);
	this->module.eval();
	std::cout << "Success.\n" << std::endl;


}
CNN::CNN(std::string model_path, std::string input_normalizer_path, std::string output_normalizer_path){
	std::cout << "*************************************************" << std::endl;
	std::cout << "Initializing CNN with given model and normalizers" << std::endl;
	std::cout << "*************************************************" << std::endl << std::endl;
	// ADD EXCEPTIOSN
	// Loading the CNN model
	std::cout << "Loading Torchscript model : "+model_path << std::endl;
	this->module = torch::jit::load(model_path);
	this->module.eval();
	std::cout << "Model loaded successfully.\n" << std::endl;

	//Load input normalizers
	std::cout << "Loading input normalizer : "+input_normalizer_path << std::endl;
	this->input_normalizer = Custom_normalizer(input_normalizer_path);
	std::cout << "Input normalizer loaded successfully.\n" << std::endl;
	
	//Load output nromalizers
	std::cout << "Loading output normalizer : "+output_normalizer_path << std::endl;
	this->output_normalizer = Custom_normalizer(output_normalizer_path);
	std::cout << "Output normlizer loaded successfully." << std::endl;
	

}

torch::Tensor CNN::predict_test(torch::Tensor input){
	// Reshape, maybe already implemented in cpp
	// Pre-processing input
	int original_height = input.size(2);
	int original_width = input.size(3);
	std::cout << "shape of input : " << input.sizes() << std::endl;
	torch::Tensor resized_input = this->resizeToNetworkSize(input);
	std::cout << "shape of resized_input : " << resized_input.sizes() << std::endl;
	torch::Tensor normalized_input = this->input_normalizer.transform(resized_input);
	std::cout << "shape of normalized_input : " << normalized_input.sizes() << std::endl;

	// Conversion Tensor to vector <IValue> for inference
	std::vector<c10::IValue> input_vector;
	input_vector.push_back(normalized_input);

	// Inference
	torch::Tensor normalized_output = this->module.forward(input_vector).toTensor();
	std::cout << "shape of normalized_output : " << normalized_output.sizes() << std::endl;

	// Post-processing
	torch::Tensor standard_output = this->output_normalizer.inverse_transform(normalized_output);
	std::cout << "shape of standard_output : " << standard_output.sizes() << std::endl;
	torch::Tensor output = this->resizeToOriginal(standard_output, std::vector<int64_t> ({original_height, original_width}));
	std::cout << "shape of output : " << output.sizes() << std::endl;
	// Resize to target grid
	// TO DO
	return output;
}

torch::Tensor CNN::predict(torch::Tensor input, float alpha){
	// Reshape, maybe already implemented in cpp
	// Pre-processing input
	int original_height = input.size(2);
	int original_width = input.size(3);
	//std::cout << "shape of input : " << input.sizes() << std::endl;
	torch::Tensor resized_input = this->resizeToNetworkSize(input);
	//std::cout << "shape of resized_input : " << resized_input.sizes() << std::endl;
	torch::Tensor normalized_input = this->input_normalizer.transform(resized_input);
	//std::cout << "shape of normalized_input : " << normalized_input.sizes() << std::endl;

	// Conversion Tensor to vector <IValue> for inference
	std::vector<c10::IValue> input_vector;
	input_vector.push_back(normalized_input);

	// Inference
	torch::Tensor normalized_output = this->module.forward(input_vector).toTensor();
	//std::cout << "shape of normalized_output : " << normalized_output.sizes() << std::endl;

	// Post-processing
	torch::Tensor standard_output = this->output_normalizer.inverse_transform(normalized_output);
	standard_output = this->set_wall_to_zero(standard_output);
	//std::cout << "shape of standard_output : " << standard_output.sizes() << std::endl;
	torch::Tensor output = this->resizeToOriginal(standard_output, std::vector<int64_t> ({original_height, original_width}));
	//std::cout << "shape of output : " << output.sizes() << std::endl;
	this->old_nut = this->current_nut;
	this->current_nut = output;
	
	//Performing under relaxation 
	//if (alpha < 1.0 && alpha > 0){
		//return this->under_relaxation(this->old_nut, this->current_nut, alpha); 
	//} else if(alpha == 0){
		//return this->old_nut;
	//} else {
	return output;
	//}
}

void CNN::save_tensor(torch::Tensor tensor_to_be_saved, std::string filename){
	torch::save(tensor_to_be_saved,filename);
}

//Foam::volScalarField CNN::convertToFoamField(torch::Tensor t,int node){
	//auto a = t.accessor<float, 1>();
	//for(int i=0; i<a.size(0); i++){
		//for (int j=0; j<a.size(1);j++){
			//volScalarField = 1.0;
		//}
	//}
//}
torch::Tensor CNN::convertToTensor_test(Foam::volVectorField& U0, Foam::volVectorField& U1){
	// Converts U0 and U1 to a torch::Tensor to be served as an input to the CNN
	const Foam::Vector<double>& a = U0.internalField()[0];
	const Foam::Vector<double>& b= U0.internalField()[1];
	const Foam::Vector<double>& c = U0.internalField()[2];
	const Foam::Vector<double>& d = U0.internalField()[3];
	// ui, vi are lines vectors --> they need to be converted as tensors
	std::cout << "Convert to Tensor : a.dim is " << a.dim << std::endl;
	std::cout << "Convert to Tensor : a.size() is " << a.size() << std::endl;
	std::cout << "Convert to Tensor : a.nCols is " << a.nCols << std::endl;
	std::cout << "Convert to Tensor : a.mRows is " << a.mRows << std::endl;
	std::cout << "Convert to Tensor : a.x() is " << a.x() << std::endl;
	std::cout << "Convert to Tensor : a.X is " << a.X << std::endl;
	std::cout << "Convert to Tensor : a.y() is " << a.y() << std::endl;
	std::cout << "Convert to Tensor : a.Y is " << a.Y << std::endl;
	std::cout << "Convert to Tensor : a.z() is " << a.z() << std::endl;
	std::cout << "Convert to Tensor : a.Z is " << a.Z << std::endl << std::endl;

	std::cout << "Convert to Tensor : b.dim is " << b.dim << std::endl;
	std::cout << "Convert to Tensor : b.size() is " << b.size() << std::endl;
	std::cout << "Convert to Tensor : b.nCols is " << b.nCols << std::endl;
	std::cout << "Convert to Tensor : b.mRows is " << b.mRows << std::endl;
	std::cout << "Convert to Tensor : b.x() is " << b.x() << std::endl;
	std::cout << "Convert to Tensor : b.X is " << b.X << std::endl;
	std::cout << "Convert to Tensor : b.y() is " << b.y() << std::endl;
	std::cout << "Convert to Tensor : b.Y is " << b.Y << std::endl;
	std::cout << "Convert to Tensor : b.z() is " << b.z() << std::endl;
	std::cout << "Convert to Tensor : b.Z is " << b.Z << std::endl << std::endl;

	std::cout << "Convert to Tensor : c.dim is " << c.dim << std::endl;
	std::cout << "Convert to Tensor : c.size() is " << c.size() << std::endl;
	std::cout << "Convert to Tensor : c.nCols is " << c.nCols << std::endl;
	std::cout << "Convert to Tensor : c.mRows is " << c.mRows << std::endl;
	std::cout << "Convert to Tensor : c.x() is " << c.x() << std::endl;
	std::cout << "Convert to Tensor : c.X is " << c.X << std::endl;
	std::cout << "Convert to Tensor : c.y() is " << c.y() << std::endl;
	std::cout << "Convert to Tensor : c.Y is " << c.Y << std::endl;
	std::cout << "Convert to Tensor : c.z() is " << c.z() << std::endl;
	std::cout << "Convert to Tensor : c.Z is " << c.Z << std::endl << std::endl;

	std::cout << "Convert to Tensor : d.dim is " << d.dim << std::endl;
	std::cout << "Convert to Tensor : d.size() is " << d.size() << std::endl;
	std::cout << "Convert to Tensor : d.nCols is " << d.nCols << std::endl;
	std::cout << "Convert to Tensor : d.mRows is " << d.mRows << std::endl;
	std::cout << "Convert to Tensor : d.x() is " << d.x() << std::endl;
	std::cout << "Convert to Tensor : d.X is " << d.X << std::endl;
	std::cout << "Convert to Tensor : d.y() is " << d.y() << std::endl;
	std::cout << "Convert to Tensor : d.Y is " << d.Y << std::endl;
	std::cout << "Convert to Tensor : d.z() is " << d.z() << std::endl;
	std::cout << "Convert to Tensor : d.Z is " << d.Z << std::endl;


	std::cout << "Convert to Tensor : U0.size() is " << U0.size()<< std::endl;
	std::cout << std::endl;

	torch::Tensor output = torch::zeros({1,4,100,250});
	//auto ouptut_a = output.accessor<double, 4>();
	int it = 0;
	forAll(U0.mesh().C(), celli){
		//Printing U0 data for testing purposes;
		std::cout << "U0 Data : ";
		std::cout << U0.mesh().C()[celli].x() << ", " << U0.mesh().C()[celli].y() << ", " << U0.mesh().C()[celli].z();
		std::cout << " Value : " << U0[celli].x() << ", " << U0[celli].y() << ", " << U0[celli].z();
		//Printing U1 data for testing purposes;
		std::cout << " U1 Data : ";
		std::cout << U1.mesh().C()[celli].x() << ", " << U1.mesh().C()[celli].y() << ", " << U1.mesh().C()[celli].z();
		std::cout << " Value : " << U1[celli].x() << ", " << U1[celli].y() << ", " << U1[celli].z();
		int i = it / 250;
		int j = it % 250;
		std::cout <<" i = " << i << ", j = " <<j;
		std::cout << std::endl;
		output[0][0][i][j] = U0.mesh().C()[celli].x(); 
		output[0][1][i][j] = U0.mesh().C()[celli].z(); 
		output[0][2][i][j] = U1.mesh().C()[celli].x(); 
		output[0][3][i][j] = U1.mesh().C()[celli].z(); 
		it++;
	}


	return output;
	
}

torch::Tensor CNN::convertToTensor(Foam::volVectorField& U0, Foam::volVectorField& U1){
	// Converts U0 and U1 to a torch::Tensor to be served as an input to the CNN

	torch::Tensor output = torch::zeros({1,4,100,250});
	//auto ouptut_a = output.accessor<double, 4>();
	int it = 0;
	forAll(U0.mesh().C(), celli){
		//Printing U0 data for testing purposes;
		//std::cout << "U0 Data : ";
		//std::cout << U0.mesh().C()[celli].x() << ", " << U0.mesh().C()[celli].y() << ", " << U0.mesh().C()[celli].z();
		//std::cout << " Value : " << U0[celli].x() << ", " << U0[celli].y() << ", " << U0[celli].z();
		////Printing U1 data for testing purposes;
		//std::cout << " U1 Data : ";
		//std::cout << U1.mesh().C()[celli].x() << ", " << U1.mesh().C()[celli].y() << ", " << U1.mesh().C()[celli].z();
		//std::cout << " Value : " << U1[celli].x() << ", " << U1[celli].y() << ", " << U1[celli].z();
		int i = it / 250;
		int j = it % 250;
		//std::cout <<" i = " << i << ", j = " <<j;
		//std::cout << std::endl;
		output[0][0][i][j] = U0[celli].x(); 
		output[0][1][i][j] = U0[celli].z(); 
		output[0][2][i][j] = U1[celli].x(); 
		output[0][3][i][j] = U1[celli].z(); 
		it++;
	}
	return output;
}

void CNN::updateFoamFieldChannelFlow(torch::Tensor &t, Foam::volScalarField &nut0, Foam::volScalarField &nut1){
	int it = 0;
	auto t_a = t.accessor<float,4>();
	forAll(nut0.mesh().C(), celli){
		int i = it / 250;
		int j = it % 250;
		nut0[celli] = t_a[0][0][i][j];
		nut1[celli] = t_a[0][1][i][j];
		it++;
	}
}

Foam::volScalarField CNN::convertToFoamField(const torch::Tensor &t, const std::string& field_name, const Foam::Time& runTime, const Foam::fvMesh& mesh, int uqNode){
	Foam::volScalarField field
		(
		 IOobject
		 (
		  field_name,
		  runTime.timeName(),
		  mesh,
		  IOobject::READ_IF_PRESENT,
		  IOobject::AUTO_WRITE
		 ),
		 mesh,
		 dimensionedScalar(field_name, dimensionSet(0,2,-2,0,0,0,0), 0.0)
		);
	auto t_a = t.accessor<float, 4>();
	int it = 0;
	forAll(field.mesh().C(), celli){
		int i = it / 250;
		int j = it % 250;
		field[celli] = t_a[0][uqNode][i][j];
	}
	return field;
}

void CNN::printFoamField(Foam::volVectorField vectorField){
	forAll(vectorField.mesh().C(), celli){
		std::cout << vectorField.name() << " Data : ";
		std::cout << vectorField.mesh().C()[celli].x() << ", " << vectorField.mesh().C()[celli].y() << ", " << vectorField.mesh().C()[celli].z();
		std::cout << " Value : " << vectorField[celli].x() << ", " << vectorField[celli].y() << ", " << vectorField[celli].z();
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void CNN::printFoamField(Foam::volScalarField scalarField){
	forAll(scalarField.mesh().C(), celli){
		std::cout << scalarField.name() << " Data : ";
		std::cout << scalarField.mesh().C()[celli].x() << ", " << scalarField.mesh().C()[celli].y() << ", " << scalarField.mesh().C()[celli].z();
		std::cout << " Value : " << scalarField[celli];
		std::cout << std::endl;
	}
	std::cout << std::endl;
}


torch::Tensor CNN::resizeToNetworkSize(const torch::Tensor &input){
	if (input.size(2)==256 && input.size(3)==128){
		std::cout << "No resizeToNetworkSize Performed" << std::endl;
		return input;
	} else {
		torch::Tensor resizedTensor = F::interpolate(input,
				F::InterpolateFuncOptions()
				.mode(torch::kBilinear)
				// MAKE SURE TO SET THE SIZE AS FLEXIBLE / AS INPUT TO the member function
				.size(std::vector<int64_t>({256, 128}))
				.align_corners(true)
				);
		return resizedTensor;
	}
}

torch::Tensor CNN::resizeToOriginal(const torch::Tensor &input, std::vector<int64_t> field_dim){
	if (input.size(2)==field_dim[0] && input.size(3)==field_dim[1]){
		std::cout << "No resizeToOriginal Performed" << std::endl;
		return input;
	} else {
		torch::Tensor resizedTensor = F::interpolate(input,
				F::InterpolateFuncOptions()
				.mode(torch::kBilinear)
				// MAKE SURE TO SET THE SIZE AS FLEXIBLE / AS INPUT TO the member function
				.size(field_dim)
				.align_corners(true)
				);
		return resizedTensor;
	}
}

torch::Tensor CNN::gaussian_smoothing(torch::Tensor input, int kernel_size, float sigma){
	torch::Tensor xcord = torch::arange(kernel_size);
	torch::Tensor x_grid = xcord.repeat(kernel_size).view({kernel_size, kernel_size});
	torch::Tensor y_grid = x_grid.t();
	torch::Tensor xy_grid = torch::stack({x_grid,y_grid}, -1);

	float mean = (float(kernel_size) -1.0) /2.0;
	float variance = pow(sigma,2);

	torch::Tensor gaussian_kernel = 1.0/(2.0*M_PI*sigma)*torch::exp(-torch::sum(torch::pow(xy_grid-mean, 2), -1)/(2.0*variance));

	// make sure the sum of the coeefs i nthee gaussian kernel equals 1
	gaussian_kernel = gaussian_kernel / torch::sum(gaussian_kernel);
	//gaussian_kernel = gaussian_kernel.view({1,1,kernel_size, kernel_size});
	//
	//auto gaussian_filter = torch::nn::Conv2d(1,1, kernel_size,1,false); # THIS LINE RETURNS CONSTRUCTOR ERROR
	//gaussian_filter->weight.data() = gaussian_kernel;
	//gaussian_filter->weight.set_requires_grad(false);
	return torch::ones({1,1});
}

torch::Tensor CNN::under_relaxation(torch::Tensor old, torch::Tensor prediction, float alpha){
	if (old.sizes() == prediction.sizes()){
		torch::Tensor used = old + alpha * (prediction - old);
		std::cout << "Under-relaxation of factor " << alpha << std::endl;
		return old;
	} else {
		std::cout << "Under relaxation error, tensors of different sizes !" << std::endl;
		std::cout << "No under-relaxation performed, returning prediction" << std::endl;
		return prediction;
	}
}

torch::Tensor CNN::under_relaxation(float alpha){
	if(alpha==1.0){
		std::cout << "Note: alpha = 1, no under relaxation preformed" << std::endl;
		return current_nut;
	}
	else if (this->old_nut.sizes() == this->current_nut.sizes()){
		torch::Tensor used = this->old_nut + alpha * (this->current_nut - this->old_nut);
		this->current_nut = used;
		return used;
	} else {
		std::cout << "Under relaxation error, tensors of different sizes !" << std::endl;
		std::cout << "No under-relaxation performed, returning prediction" << std::endl;
		return this->current_nut;
	}
}


float CNN::get_max_rate_of_change(){
	//torch::Tensor max_nut0_old = torch::max(torch::max(torch::squeeze(this->old_nut).index({0,Slice(None),Slice(None)})));
	//torch::Tensor max_nut1_old = torch::max(torch::max(torch::squeeze(this->old_nut).index({1,Slice(None),Slice(None)})));
	//torch::Tensor max_nut0_current = torch::max(torch::max(torch::squeeze(this->current_nut).index({0,Slice(None),Slice(None)})));
	//torch::Tensor max_nut1_current = torch::max(torch::max(torch::squeeze(this->current_nut).index({1,Slice(None),Slice(None)})));
//
	//torch::Tensor min_nut0_old = torch::min(torch::min(torch::squeeze(this->old_nut).index({0,Slice(None),Slice(None)})));
	//torch::Tensor min_nut1_old = torch::min(torch::min(torch::squeeze(this->old_nut).index({1,Slice(None),Slice(None)})));
	//torch::Tensor min_nut0_current = torch::min(torch::min(torch::squeeze(this->current_nut).index({0,Slice(None),Slice(None)})));
	//torch::Tensor min_nut1_current = torch::min(torch::min(torch::squeeze(this->current_nut).index({1,Slice(None),Slice(None)})));
//
	//torch::Tensor diff_max0 = torch::abs((max_nut0_old - max_nut0_current)/max_nut0_old);
	//torch::Tensor diff_max1 = torch::abs(max_nut1_old - max_nut1_current);
	return 0.0;

}

torch::Tensor CNN::set_wall_to_zero(const torch::Tensor &input){
	torch::Tensor output = input;
	//auto output_a = output.accessor<float, 2>();
	for(int j = 0;j<output.sizes()[3];j++){
		output[0][0][0][j]=0.0;
		output[0][0][1][j]=0.0;
	}
	return output;
}
