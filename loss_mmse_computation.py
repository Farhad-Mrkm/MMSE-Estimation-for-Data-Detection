

h = 1.0
def k_computation(data):
    #x = np.linspace(-100, 100, 128)
   # y = np.linspace(-100, 100, 128)

   # xv, yv = np.meshgrid(x, y)
  

   # grid_data = np.stack([xv.flatten(), yv.flatten()], axis=-1)
   # grid_data1=grid_data[:16256,:]
    z_joint, logdet_joint = joint_model(data)
    log_prob_x_joint = joint_model.distribution.log_prob(z_joint) + logdet_joint
    prob_x_joint = tf.exp(log_prob_x_joint)
    
    z_marginal, logdet_marginal = marginal_model(data)
    log_prob_x_marginal = marginal_model.distribution.log_prob(z_marginal) + logdet_marginal
    prob_x_marginal = tf.exp(log_prob_x_marginal)
    
    k_proportion = prob_x_marginal / (prob_x_joint + 1e-8)
    return k_proportion
def mmse_loss(joint_data, marginal_data):
    
   # k_proportion=.5*tf.exp(-(marginal_data[:,0]**2+.5*marginal_data[:,1]**2-2*marginal_data[:,0]*marginal_data[:,1]))
    k_proportion=k_computation(marginal_data)
    mmse_model_output_joint = tf.squeeze(mmse_model(joint_data), axis=-1)
    mmse_model_output_marginal = tf.exp(tf.squeeze(mmse_model(marginal_data), axis=-1))
    
    
    first_term = tf.reduce_mean(mmse_model_output_joint)
    second_term = tf.reduce_mean(k_proportion * mmse_model_output_marginal / tf.reduce_mean(mmse_model_output_marginal))
    
    loss = (1 / h) * (first_term - second_term)

    
    return tf.abs(loss),k_proportion