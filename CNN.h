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

/** @brief Class to simplify the use of NN models in C++, which also contains functions to convert OpenFOAM meshing to raw tensor fields and vice versa. 

    Supports only the use of models saved under the Torchscript format (PyTorch) 
    @author Maheindrane Many
    @date March 2022
    */

class CNN {

	torch::jit::script::Module module;
	Custom_normalizer input_normalizer;
	Custom_normalizer output_normalizer;

	// For under_relaxation
	torch::Tensor old_nut = torch::zeros(1);
	torch::Tensor current_nut = torch::zeros(0);


	public:



		// Constructors 
		/** Default constructor to initialize the CNN object with model only (without normalizers)
		@param model_path string which contains the path of the model in Torchscript format.
        	*/
		CNN(std::string model_path);
		/** Constructor recommended to use to initialize the CNN object with model and normalizers
		@param model_path string which contains the path of the model in Torchscript format.
		@param input_normalizer_path string which contains the absolute path to the input normalizer in JSON format, the normalizers are used to create objects from the Custom_Normalizer class.
		@param input_normalizer_path string which contains the absolute path to the output normalizer in JSON format (See Custom_Normalizer class).
        	*/
		CNN(std::string model_path, std::string input_normalizer_path, std::string output_normalizer_path);
		/** Default destructor of the CNN class
        	*/
		~CNN();


		// Member functions
		/** Main function to perform inference with the model specified during initialization of the class.
		 * Input is first resized to the size of the CNN using bilinear interpolation
		 * Normalized with the normalizers specified during initialization
		 * Inference is performed
		 * The resulting output is denormalised and a resized back to the original tensor size.
		 * The latest prediction (torch::Tensor) is stored in the attribute current_nut
		@param input torch::Tensor used as an input for inference, dimensions of the tensors are (1(samples),4(channels),:(height),:(width))
		@return torch::Tensor
        	*/
		torch::Tensor predict(torch::Tensor input); // alpha is the under-relaxation factor (1 == no UR ; 0 == no prediction)
		/** Same function as CNN::predict with the difference that a mask is used as an additional channel 
		@param input torch::Tensor used as an input for inference, a geometry mask is added (total of 5 channels)
		@return torch::Tensor
        	*/
		torch::Tensor predict_mask(torch::Tensor input);
		torch::Tensor predict_test(torch::Tensor input); // For in-between shape printing
		/** Utility function used to save a specific tensor using the torchscript notation
		@param tensor_to_be_saved Tensor to be saved.
		@param filename where absolute path with filename where the tensor is to be saved.
        	*/
		void save_tensor(torch::Tensor tensor_to_be_saved, std::string filename);


		//For OF field conversion to torch::Tensor
		torch::Tensor convertToTensor_test(Foam::volVectorField& U0, Foam::volVectorField& U1);
		/** Extrtacts the lengthwise and widthwise \f$u_x, u_y\f$ velocities from the vector fields U0 and U1 and stores them as tensors.
		 * Only works for a rectangular domain with uniform meshing, dimensions of the mesh have to introduced separately.
		 * The produced tensor is of the folloowing shape : [1,4,width_discretizations, mesh length discretizations].
		 * The 4 channels are order in the following manner U0_x, U0_y, U1_x, U1_y 
		@param U0 volVectorField, first velocity mode
		@param U1 volVectorField, second velocity mode
		@return torch::Tensor
        	*/
		torch::Tensor convertToTensor(Foam::volVectorField& U0, Foam::volVectorField& U1);
		/** Extrtacts the lengthwise and widthwise \f$u_x, u_y\f$ velocities from the vector fields U0 and U1 and stores them as tensors.
		 * Adapted for a Backward facing step configuration, dimensions of the mesh have to coded separately.
		 * The produced tensor is of the folloowing shape : [1,4,width_discretizations, mesh length discretizations].
		 * The 4 channels are order in the following manner U0_x, U0_y, U1_x, U1_y 
		@param U0 volVectorField, first velocity mode
		@param U1 volVectorField, second velocity mode
		@return torch::Tensor
        	*/
		torch::Tensor convertToTensor_bfs(Foam::volVectorField& U0, Foam::volVectorField& U1);
		/** Extrtacts the lengthwise and widthwise \f$u_x, u_y\f$ velocities from the vector fields U0 and U1 and stores them as tensors.
		 * Adapted for a Backward facing step configuration, dimensions of the mesh have to coded separately.
		 * The emulated backward facing step geometry is added as a n additional channel using the define_mask() function from the CNN class.
		 * The produced tensor is of the folloowing shape : [1,5,width_discretizations, mesh length discretizations].
		 * The 5 channels are ordered in the following manner: U0_x, U0_y, U1_x, U1_y 
		@param U0 volVectorField, first velocity mode
		@param U1 volVectorField, second velocity mode
		@return torch::Tensor
        	*/
		torch::Tensor convertToTensor_bfs_mask(Foam::volVectorField& U0, Foam::volVectorField& U1);
		/** Updates the existing nut0 and nut1 Scalar fields by assigning the values from the tensor t.
		@param t tensor of the same dimensions as the scalar fields 
		@param nut0 volScalarField to update with the first channel of the tensor
		@param nut1 volScalarField to update with the second channel of the tensor
		
        	*/
		void updateFoamFieldChannelFlow(torch::Tensor& t, Foam::volScalarField& nut0, Foam::volScalarField& nut1);
		/** performs the same operation as updateFoamFieldChannelFlow, but for an emulated backward facing step configuration.
		@param t  
		@param nut0 volScalarField to update with the first channel of the tensor
		@param nut1 volScalarField to update with the second channel of the tensor
		
        	*/
		void updateFoamFieldChannelFlow_bfs(torch::Tensor& t, Foam::volScalarField& nut0, Foam::volScalarField& nut1);
		/** Testing function
		
        	*/
		void updateFoamFieldChannelFlow_velocity(torch::Tensor& t, Foam::volVectorField& U0, Foam::volVectorField& U1);
		/** Creates a new Scalar field with the information on the tensor t
		@param t tensor whose data is set in the scalar field 
		@param field_name name of the the newly created volScalarField
		@param runTime relative to OpenFOAM
		@param mesh relative to OpenFOAM
		@param uqNode needed to specify which node it corresponds to. In case of multiple turbulence model solved separately this is needed.
        	*/
		Foam::volScalarField convertToFoamField(const torch::Tensor &t, const std::string& field_name, const Foam::Time& runTime, const Foam::fvMesh& mesh, int uqNode);


		// Output transforms (smoothing etc...)
		/** Transform function to perform gaussian smoothing of the input tensor
		 * OBSOLETE
		@param input tensor to be smoothed
		@param kernel_size filter_size used in the gaussian smoothing	
		@param sigma parameter
        	*/
		torch::Tensor gaussian_smoothing(torch::Tensor input, int kernel_size, float sigma);
		/** Performs under-relaxation using the specified under-relaxation factor
		 * Operation performed is : output = this->old_nut + alpha* (this->current_nut + this->old_nut)
		@param alpha under-relaxation factor
        	*/
		torch::Tensor under_relaxation(float alpha);
		/** Performs under-relaxation using the specified under-relaxation factor and the specified Tensor
		 * Operation performed is : output = old + alpha* (current - old)
		@param alpha under-relaxation factor
		@param old tensor to be added with a fraction of "current" tensor
		@param current tensor to be under-relaxed
        	*/
		torch::Tensor under_relaxation(torch::Tensor old, torch::Tensor current, float alpha);


		// Resize tensor functions
		//
		/** Resize the given tensor to the input size of the Convolutional Neural Network. The input size of the CNN has to be directly modified in the code. Member function to be implemented further. The resizing operation is bilinear interpolation.
		@param input tensor to be resized
		@return tensor with adapted last two dimensions.
        	*/
		torch::Tensor resizeToNetworkSize(const torch::Tensor& input);
		/** Resizes the tensor input to the specified size
		@param input tensor to be resized
		@param size vector which contains the dimensions of the final tensor ({1,4,height,width})
		@return tensor with adapted last two dimensions.
        	*/
		torch::Tensor resizeToOriginal(const torch::Tensor& input,std::vector<int64_t> size);


		//Utility member functions
		/**  (Not tested yet) Experimental function which is aimed to give information on the max rate of change of the CNN predictions. If such a function is working, adaptive under-relaxation can be performed when the maximum rate of change of the prediction compared to the previous prediction exceeds a certain value. 
		@return a float value
        	*/
		float get_max_rate_of_change();
		/**  Takes a tensor and sets the value at the wall to zero.
		 @param input tensor to be processes
		@return returns a tensor.
        	*/
		torch::Tensor set_wall_to_zero(const torch::Tensor& input);
		/**  Sets the value at the boundary  patch called "inlet" by a constant value. Useful to manually initialize the value at the boundary faces. In the thesis, this is used to initialize the value of the turbulent viscosity field at a specific value.
		 @param  field Scalar field whose inlet patch is to be processed.
		 @param value float value used to set the inlet patch
        	*/
		void set_inlet_nut(Foam::volScalarField& field, const float value);
		/**  Overload of the set_inlet_nut() function for volVectorFields. Sets the value at the boundary  patch called "inlet" by a constant value. Useful to manually initialize the value at the boundary faces. In the thesis, this is used to initialize the first mode of the turbulent viscosity field at a specific value.
		 @param  field Vector Field
		 @param value float value used to set the inlet patch
        	*/
		void set_inlet_nut(Foam::volVectorField& field, const float value);
		/**  Utility function used to generate a mask of the geometry. At the moment, only an emulated backward facing step is supported.
		 @param  mask_pixel_height total height discretizations of the mesh 
		 @param  mask_pixel_width total width discretizations of the mesh 
		 @param  step_height height of the step in meters.
		 @param  domain_height height of the domain in meters.
		 @param  step_width width of the step in meters.
		 @param  domain_width width of the domain in meters.
        	*/
		torch::Tensor define_mask(int mask_pixel_height, int mask_pixel_width, float step_height, float domain_height, float step_width, float domain_width);
		
		 
		// Printing Foam Fields
		/** The print* functions are experimental functions that were needed to understand the looping proceduree over the cells of the mesh. They are pure utility functions.
		 *
		 */
		void printFoamField(const Foam::volVectorField& vectorField);
		void printFoamField(const Foam::volScalarField& scalarField);
		void printFoamFieldNodes(const Foam::volVectorField& vectorField);
		void printFoamFieldNodes(const Foam::volScalarField& scalarField);
 		void printInletNodes(const Foam::volScalarField& scalarField); // For testing
		void printInletNodesBis(const Foam::volScalarField& scalarField);

		// Loading tensor in a file
		/** Allows to load a tensor stored in a container (similar to a dictionnary) stored using Torchscript. This explains the process well, https://shuye.dev/kb/AI-engineering/LibTorch 
		 @param container_filepath path to the container (string)
		 @param key the tensor to be loaded is saved under a certain key. This is the key (string). 
		 */
		torch::Tensor loadTensorFromContainer(const std::string container_filepath, const std::string key);
};
