# NeuralMMSE
 source code for "Neural MMSE Estimation for Data detection"
 
 
 How to run:
 
 First part is for performing the density evaluation:
 
 
 
 
 1- produce the data (inputs and noisy version of them) using Gen_data
 
 
 2- use joint_marginal_batch with auto option to produce samples of marginal and joint distributions.
 
 
 3-Run real_nvp using joint and marginal data samples to acquire an estimate of their distribution
 
 
 Second part is for computing the optimal MMSE:
 
 
 
 
 1-Run MMSE_model
 
 
 2-Using function joint_marginal_batch with option other than 'auto' produce the customized data
 
 
 3-Run the loss_MMSE_computation for computing 'k' and MMSE loss.
 
 
 4-Use MMSE_trainer to traint the neural_network for computing the optimal MMSE value.
