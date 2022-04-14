#include "InferenceEngine.h"
#include "ATen/Functions.h"
#include "fvBoundaryMesh.H"
#include "labelList.H"
#include "torch/nn/modules/conv.h"
#include "vectorField.H"
#include "volFieldsFwd.H"
#include <cmath>
//#include "UList.H"
//#include <iostream>
//#include "volFieldsFwd.H"
//#include <ATen/core/ivalue.h>
//#include <torch/serialize.h>
//#include <vector
namespace F = torch::nn::functional;

InferenceEngine::InferenceEngine(std::string model_path){
	std::cout << "*************************************************" << std::endl;
	std::cout << "-------Initializing InferenceEngine with given model --------" << std::endl;
	std::cout << "*************************************************" << std::endl << std::endl;
	// ADD EXCEPTIOSN
	// Loading the InferenceEngine model
	std::cout << "Loading Torchscript model : "+model_path << std::endl;
	this->module = torch::jit::load(model_path);
	this->module.eval();
	std::cout << "Success.\n" << std::endl;


}
InferenceEngine::InferenceEngine(std::string model_path, std::string input_normalizer_path, std::string output_normalizer_path){
	std::cout << "*************************************************" << std::endl;
	std::cout << "Initializing InferenceEngine with given model and normalizers" << std::endl;
	std::cout << "*************************************************" << std::endl << std::endl;
	// ADD EXCEPTIOSN
	// Loading the InferenceEngine model
	std::cout << "Loading Torchscript model : "+model_path << std::endl;
	this->module = torch::jit::load(model_path);
	this->module.eval();
	std::cout << "Model loaded successfully.\n" << std::endl;

	//Load input normalizers
	std::cout << "Loading input normalizer : "+input_normalizer_path << std::endl;
	this->input_normalizer = CustomNormalizer(input_normalizer_path);
	std::cout << "Input normalizer loaded successfully.\n" << std::endl;
	
	//Load output nromalizers
	std::cout << "Loading output normalizer : "+output_normalizer_path << std::endl;
	this->output_normalizer = CustomNormalizer(output_normalizer_path);
	std::cout << "Output normlizer loaded successfully." << std::endl;
	

}

torch::Tensor InferenceEngine::predict_test(torch::Tensor input){
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

torch::Tensor InferenceEngine::predict(torch::Tensor input){
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
	//standard_output = this->set_wall_to_zero(standard_output);
	//std::cout << "shape of standard_output : " << standard_output.sizes() << std::endl;
	torch::Tensor output = this->resizeToOriginal(standard_output, std::vector<int64_t> ({original_height, original_width}));
	//std::cout << "shape of output : " << output.sizes() << std::endl;
	this->old_nut = this->current_nut;
	this->current_nut = output;
	
	return output;
	//}
}
torch::Tensor InferenceEngine::predict_mask(torch::Tensor input){
	// Reshape, maybe already implemented in cpp
	// Pre-processing input
	int original_height = input.size(2);
	int original_width = input.size(3);
	//std::cout << "shape of input : " << input.sizes() << std::endl;
	torch::Tensor resized_input = this->resizeToNetworkSize(input);
	//std::cout << "shape of resized_input : " << resized_input.sizes() << std::endl;
	torch::Tensor normalized_input = this->input_normalizer.transform(resized_input);
	//std::cout << "shape of normalized_input : " << normalized_input.sizes() << std::endl;
	
	//Adding mask
	torch::Tensor mask = this->define_mask(normalized_input.size(2),normalized_input.size(3),1.0,3.0,2.0,17.0);
	normalized_input = torch::cat({normalized_input, mask}, 1);
	

	// Conversion Tensor to vector <IValue> for inference
	std::vector<c10::IValue> input_vector;
	input_vector.push_back(normalized_input);

	// Inference
	torch::Tensor normalized_output = this->module.forward(input_vector).toTensor();
	//std::cout << "shape of normalized_output : " << normalized_output.sizes() << std::endl;

	// Post-processing
	torch::Tensor standard_output = this->output_normalizer.inverse_transform(normalized_output);
	//standard_output = this->set_wall_to_zero(standard_output);
	//std::cout << "shape of standard_output : " << standard_output.sizes() << std::endl;
	torch::Tensor output = this->resizeToOriginal(standard_output, std::vector<int64_t> ({original_height, original_width}));
	//std::cout << "shape of output : " << output.sizes() << std::endl;
	this->old_nut = this->current_nut;
	this->current_nut = output;
	
	//Performing under relaxation 
	return output;
	//}
}

void InferenceEngine::save_tensor(torch::Tensor tensor_to_be_saved, std::string filename){
	torch::save(tensor_to_be_saved,filename);
}

//Foam::volScalarField InferenceEngine::convertToFoamField(torch::Tensor t,int node){
	//auto a = t.accessor<float, 1>();
	//for(int i=0; i<a.size(0); i++){
		//for (int j=0; j<a.size(1);j++){
			//volScalarField = 1.0;
		//}
	//}
//}
torch::Tensor InferenceEngine::convertToTensor_test(Foam::volVectorField& U0, Foam::volVectorField& U1){
	// Converts U0 and U1 to a torch::Tensor to be served as an input to the InferenceEngine
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

torch::Tensor InferenceEngine::convertToTensor(Foam::volVectorField& U0, Foam::volVectorField& U1){
	// Converts U0 and U1 to a torch::Tensor to be served as an input to the InferenceEngine

	torch::Tensor output = torch::zeros({1,4,100,50});
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
		int i = it / 50;
		int j = it % 50;
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


void InferenceEngine::updateFoamFieldChannelFlow(torch::Tensor &t, Foam::volScalarField &nut0, Foam::volScalarField &nut1){
	int it = 0;
	auto t_a = t.accessor<float,4>();
	forAll(nut0.mesh().C(), celli){
		int i = it / 50;
		int j = it % 50;
		nut0[celli] = t_a[0][0][i][j];
		nut1[celli] = t_a[0][1][i][j];
		it++;
	}
}

torch::Tensor InferenceEngine::convertToTensor_bfs(Foam::volVectorField& U0, Foam::volVectorField& U1){
	// Converts U0 and U1 to a torch::Tensor to be served as an input to the InferenceEngine

	//Number of cells definitions
	//Step height
	int nsh=33;
	//domain height-step height
	int nh=67;
	//Step length
	int nsl=6;
	// domain length - step length
	int nl=44;

	int height_disc=100;
	int width_disc=50;

	torch::Tensor output = torch::zeros({1,4,height_disc,width_disc});
	int it = 0;
	forAll(U0.mesh().C(), celli){
		if(it<nsl*nh){
			int i = it / nsl;
			int j = it % nsl;
			output[0][0][nsh+i][j] = U0[celli].x(); 
			output[0][1][nsh+i][j] = U0[celli].z(); 
			output[0][2][nsh+i][j] = U1[celli].x(); 
			output[0][3][nsh+i][j] = U1[celli].z(); 
		}
		else if (it<(nsl*nh+nsh*nsl)){
			int i = (it-nh*nsl) / nsl;
			int j = (it-nh*nsl) % nsl;
			output[0][0][i][j] = U0[celli].x(); 
			output[0][1][i][j] = U0[celli].z(); 
			output[0][2][i][j] = U1[celli].x(); 
			output[0][3][i][j] = U1[celli].z(); 
		}
		else if (it<(nsl*nh+nsh*nsl+nh*nl)){
			int i = (it-(nh*nsl+nsh*nsl)) / nl;
			int j = (it-(nh*nsl+nsh*nsl)) % nl;
			output[0][0][i+nsh][j+nsl] = U0[celli].x(); 
			output[0][1][i+nsh][j+nsl] = U0[celli].z(); 
			output[0][2][i+nsh][j+nsl] = U1[celli].x(); 
			output[0][3][i+nsh][j+nsl] = U1[celli].z(); 
		}
		else if (it<(nsl*nh+nsh*nsl+nh*nl+nsh*nl)){
			int i = (it-(nh*nsl+nsh*nsl+nh*nl)) / nl;
			int j = (it-(nh*nsl+nsh*nsl+nh*nl)) % nl;
			output[0][0][i][j+nsl] = U0[celli].x(); 
			output[0][1][i][j+nsl] = U0[celli].z(); 
			output[0][2][i][j+nsl] = U1[celli].x(); 
			output[0][3][i][j+nsl] = U1[celli].z(); 
		}
		it++;
	}
	return output;
}

torch::Tensor InferenceEngine::convertToTensor_bfs_mask(Foam::volVectorField& U0, Foam::volVectorField& U1){
	// Converts U0 and U1 to a torch::Tensor to be served as an input to the InferenceEngine

	//Number of cells definitions
	//Step height
	int nsh=33;
	//domain height-step height
	int nh=67;
	//Step length
	int nsl=6;
	// domain length - step length
	int nl=44;

	int height_disc=100;
	int width_disc=50;

	torch::Tensor output = torch::zeros({1,5,height_disc,width_disc});
	int it = 0;
	forAll(U0.mesh().C(), celli){
		if(it<nsl*nh){
			int i = it / nsl;
			int j = it % nsl;
			output[0][0][nsh+i][j] = U0[celli].x(); 
			output[0][1][nsh+i][j] = U0[celli].z(); 
			output[0][2][nsh+i][j] = U1[celli].x(); 
			output[0][3][nsh+i][j] = U1[celli].z(); 
		}
		else if (it<(nsl*nh+nsh*nsl)){
			int i = (it-nh*nsl) / nsl;
			int j = (it-nh*nsl) % nsl;
			output[0][0][i][j] = U0[celli].x(); 
			output[0][1][i][j] = U0[celli].z(); 
			output[0][2][i][j] = U1[celli].x(); 
			output[0][3][i][j] = U1[celli].z(); 
		}
		else if (it<(nsl*nh+nsh*nsl+nh*nl)){
			int i = (it-(nh*nsl+nsh*nsl)) / nl;
			int j = (it-(nh*nsl+nsh*nsl)) % nl;
			output[0][0][i+nsh][j+nsl] = U0[celli].x(); 
			output[0][1][i+nsh][j+nsl] = U0[celli].z(); 
			output[0][2][i+nsh][j+nsl] = U1[celli].x(); 
			output[0][3][i+nsh][j+nsl] = U1[celli].z(); 
		}
		else if (it<(nsl*nh+nsh*nsl+nh*nl+nsh*nl)){
			int i = (it-(nh*nsl+nsh*nsl+nh*nl)) / nl;
			int j = (it-(nh*nsl+nsh*nsl+nh*nl)) % nl;
			output[0][0][i][j+nsl] = U0[celli].x(); 
			output[0][1][i][j+nsl] = U0[celli].z(); 
			output[0][2][i][j+nsl] = U1[celli].x(); 
			output[0][3][i][j+nsl] = U1[celli].z(); 
		}
		it++;
	}
	torch::Tensor mask = this->define_mask(height_disc,width_disc,1.0,3.0,2.0,17.0);
	output.index_put_({ 0, 4, torch::indexing::Slice(), torch::indexing::Slice() }, mask);
	return output;
}


torch::Tensor InferenceEngine::define_mask(int mask_pixel_height, int mask_pixel_width, float step_height, float domain_height, float step_width, float domain_width){
	torch::Tensor mask = torch::ones({1,1,mask_pixel_height, mask_pixel_width});
	int separationindexx = std::floor(step_width/domain_width*mask_pixel_width);
	int separationindexy = std::floor(step_height/domain_height*mask_pixel_height);
	for (int i=0; i<mask_pixel_height; ++i){
		for (int j=0; j<mask_pixel_width; ++j){
			if (i<= separationindexy && j<=separationindexx){
				mask[0][0][i][j] = 0;
			}
		}
	}
	return mask;
}

void InferenceEngine::updateFoamFieldChannelFlow_bfs(torch::Tensor &t, Foam::volScalarField &nut0, Foam::volScalarField &nut1){
	int it = 0;
	//Number of cells definitions
	//Step height
	int nsh=33;
	//domain height-step height
	int nh=67;
	//Step length
	int nsl=6;
	// domain length - step length
	int nl=44;

	auto t_a = t.accessor<float,4>();
	forAll(nut0.mesh().C(), celli){
		if(it<nsl*nh){
			int i = it / nsl;
			int j = it % nsl;
			nut0[celli] = t_a[0][0][nsh+i][j] ; 
			nut1[celli] = t_a[0][1][nsh+i][j] ; 
		}
		else if (it<(nsl*nh+nsh*nsl)){
			int i = (it-nh*nsl) / nsl;
			int j = (it-nh*nsl) % nsl;
			nut0[celli] = t_a[0][0][i][j] ; 
			nut1[celli] = t_a[0][1][i][j] ; 
		}
		else if (it<(nsl*nh+nsh*nsl+nh*nl)){
			int i = (it-(nh*nsl+nsh*nsl)) / nl;
			int j = (it-(nh*nsl+nsh*nsl)) % nl;
			nut0[celli] = t_a[0][0][i+nsh][j+nsl] ; 
			nut1[celli] = t_a[0][1][i+nsh][j+nsl] ; 
		}
		else if (it<(nsl*nh+nsh*nsl+nh*nl+nsh*nl)){
			int i = (it-(nh*nsl+nsh*nsl+nh*nl)) / nl;
			int j = (it-(nh*nsl+nsh*nsl+nh*nl)) % nl;
			nut0[celli] = t_a[0][0][i][j+nsl] ; 
			nut1[celli] = t_a[0][1][i][j+nsl] ; 
		}
		it++;
	}
}

Foam::volScalarField InferenceEngine::convertToFoamField(const torch::Tensor &t, const std::string& field_name, const Foam::Time& runTime, const Foam::fvMesh& mesh, int uqNode){
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
		int i = it / 50;
		int j = it % 50;
		field[celli] = t_a[0][uqNode][i][j];
	}
	return field;
}

void InferenceEngine::printFoamField(const Foam::volVectorField& vectorField){
	forAll(vectorField.mesh().C(), celli){
		std::cout << vectorField.name() << " Data : ";
		std::cout << vectorField.mesh().C()[celli].x() << ", " << vectorField.mesh().C()[celli].y() << ", " << vectorField.mesh().C()[celli].z();
		std::cout << " Value : " << vectorField[celli].x() << ", " << vectorField[celli].y() << ", " << vectorField[celli].z();
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void InferenceEngine::printFoamField(const Foam::volScalarField& scalarField){
	forAll(scalarField.mesh().C(), celli){
		std::cout << scalarField.name() << " Data : ";
		std::cout << scalarField.mesh().C()[celli].x() << ", " << scalarField.mesh().C()[celli].y() << ", " << scalarField.mesh().C()[celli].z();
		std::cout << " Value : " << scalarField[celli];
		std::cout << std::endl;
	}
	std::cout << std::endl;
}


torch::Tensor InferenceEngine::resizeToNetworkSize(const torch::Tensor &input){
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

torch::Tensor InferenceEngine::resizeToOriginal(const torch::Tensor &input, std::vector<int64_t> field_dim){
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

torch::Tensor InferenceEngine::gaussian_smoothing(torch::Tensor input, int kernel_size, float sigma){
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

torch::Tensor InferenceEngine::under_relaxation(torch::Tensor old, torch::Tensor prediction, float alpha){
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

torch::Tensor InferenceEngine::under_relaxation(float alpha){
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


float InferenceEngine::get_max_rate_of_change(){
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

torch::Tensor InferenceEngine::set_wall_to_zero(const torch::Tensor &input){
	torch::Tensor output = input;
	//auto output_a = output.accessor<float, 2>();
	for(int j = 0;j<output.sizes()[3];j++){
		output[0][0][0][j]=0.0;
		output[0][0][-1][j]=0.0;
	}
	return output;
}

void InferenceEngine::printFoamFieldNodes(const Foam::volVectorField& vectorField){
	std::cout << "Printing Node data of Field" << std::endl;
	const Foam::labelListList& points = vectorField.mesh().cellPoints();
	forAll(points, celli){
		std::cout << vectorField.name() << " Data : ";
		// std::cout << points[celli].begin << ", " << points[celli].y() << ", " << points[celli].z();
		forAll(points[celli],i){
			std::cout << i << ":" << points[celli][i] << ", ";
		}
		// std::cout << points[celli] << std::endl;
		// std::cout << " Value : " << vectorField[celli].x() << ", " << vectorField[celli].y() << ", " << vectorField[celli].z();
		std::cout << std::endl;
	}
	std::cout << std::endl;

	forAll(points, celli){
		std::cout << vectorField.name() << " Data : ";
		// std::cout << points[celli].begin << ", " << points[celli].y() << ", " << points[celli].z();
		forAll(points[celli],i){
			int current_node = points[celli][i];
			std::cout << i << ":" << current_node << "_value0:" << vectorField.mesh().points()[current_node].component(0);
			std::cout << "_value1:" << vectorField.mesh().points()[current_node].component(1);
			std::cout << "_value2:" << vectorField.mesh().points()[current_node].component(2);
		}
		// std::cout << points[celli] << std::endl;
		// std::cout << " Value : " << vectorField[celli].x() << ", " << vectorField[celli].y() << ", " << vectorField[celli].z();
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void InferenceEngine::printFoamFieldNodes(const Foam::volScalarField& scalarField){
	std::cout << "Support for scalarFields is to be implemented" << std::endl;
	std::cout << std::endl;
}

void InferenceEngine::printInletNodes(const Foam::volScalarField& scalarField){
	std::cout << "Printing Inlet nodes" << std::endl;
	Foam::label patchID = scalarField.mesh().boundaryMesh().findPatchID("inlet");
	// std::cout << patchID << endl;
	// auto boundary = scalarField.mesh().boundaryMesh()["inlet"];
	auto face_centres_boundary = scalarField.mesh().Cf().boundaryField()[patchID];
	forAll(face_centres_boundary, facei){
		// std::cout << "value: " << face_centres_boundary[facei] << ", x:"<< face_centres_boundary[facei].x() << ", y:"<< face_centres_boundary[facei].y() <<", z:"<< face_centres_boundary[facei].z() << std::endl;
	}
	std::cout << std::endl;
}

void InferenceEngine::printInletNodesBis(const Foam::volScalarField& scalarField){
	std::cout << "Printing Inlet nodes" << std::endl;
	Foam::label patchID = scalarField.mesh().boundaryMesh().findPatchID("inlet");
	auto points_ = scalarField.mesh().points();

	forAll (scalarField.mesh().boundary()[patchID],facei) 
	{
		const label& faceID = scalarField.mesh().boundaryMesh()[patchID].start() + facei;
		forAll (scalarField.mesh().faces()[faceID], nodei)
		{
			const label& nodeID = scalarField.mesh().faces()[faceID][nodei];
			std::cout << points_[nodeID].X << std::endl;
		}
	}
	// std::cout << patchID << endl;
	// auto boundary = scalarField.mesh().boundaryMesh()["inlet"];
	auto face_centres_boundary = scalarField.mesh().Cf().boundaryField()[patchID];
	forAll(face_centres_boundary, facei){
		// std::cout << "value: " << face_centres_boundary[facei] << ", x:"<< face_centres_boundary[facei].x() << ", y:"<< face_centres_boundary[facei].y() <<", z:"<< face_centres_boundary[facei].z() << std::endl;
	}
	std::cout << std::endl;
}

torch::Tensor InferenceEngine::loadTensorFromContainer(const std::string container_filepath, const std::string key){
	std::cout << "Note : to load Tensors from the Pytorch Python API, they need to be saved under a container with corresponding dictionnary keys. See sample code to save Pytorch Python Tensors as .pt containers."<< std::endl;
	std::cout << "Loading Container located : " << container_filepath << std::endl;
	std::cout << "Loading Key from Container located : " << key << std::endl;
	torch::jit::script::Module container = torch::jit::load(container_filepath);
	// Load values by name
	torch::Tensor loaded_tens = container.attr(key).toTensor();
	return loaded_tens;
}

void InferenceEngine::updateFoamFieldChannelFlow_velocity(torch::Tensor& t, Foam::volVectorField& U0, Foam::volVectorField& U1){
	int it = 0;
	auto t_a = t.accessor<float,4>();
	forAll(U0.mesh().C(), celli){
		int i = it / 50;
		int j = it % 50;
		std::cout << " Changed : x:"<<U0[celli].component(0)<<", y:"<<U0[celli].component(1)<<", z:"<<U0[celli].component(2)<<std::endl;
		U0[celli].component(0) = t_a[0][0][i][j];
		U0[celli].component(2) = t_a[0][1][i][j];
		U1[celli].component(0) = t_a[0][2][i][j];
		U1[celli].component(2) = t_a[0][3][i][j];
		it++;
	}
}
//void InferenceEngine::set_inlet_nut(Foam::volScalarField& field, float value){
//	Foam::label patchID = field.mesh().boundaryMesh().findPatchID("inlet");
//	const fvPatch& boundaryPatch = field.mesh().boundary()[patchID];
//
//	forAll(boundaryPatch, faceI)
//	{
//		field[faceI] =  value;
//	}
//}

void InferenceEngine::set_inlet_nut(Foam::volVectorField& field, float value){
	Foam::label patchID = field.mesh().boundaryMesh().findPatchID("inlet");
	const fvPatch& boundaryPatch = field.mesh().boundary()[patchID];

	forAll(boundaryPatch, faceI)
	{
		field[faceI] =  vector(value,value,value);
	}
}

void InferenceEngine::set_inlet_nut(Foam::volScalarField& field, float value){
	label patchI = field.mesh().boundaryMesh().findPatchID("inlet");
	std::cout << "found \"inlet\" patch ID : "<< patchI << std::endl;

	//forAll(field.mesh().boundaryMesh()[patchI].faceCentres(), faceI)
	//{
	//	scalar x = field.mesh().boundaryMesh()[patchI].faceCentres()[faceI].x();
	//	scalar y = field.mesh().boundaryMesh()[patchI].faceCentres()[faceI].y();
	//	scalar z = field.mesh().boundaryMesh()[patchI].faceCentres()[faceI].z();
	//	std::cout<<faceI<<" "<<x<<" "<<y<<" "<<z<<" "<<endl;
	//}

	forAll(field.boundaryFieldRef()[patchI], faceIt){
		std::cout << field.boundaryFieldRef()[patchI][faceIt] << std::endl;
		field.boundaryFieldRef()[patchI][faceIt] = value;

		std::cout << field.boundaryFieldRef()[patchI][faceIt] << std::endl;
	}
}
