



def joint_marginal_batch(x, y,snr=10,loop_range=1,mode='auto',inptype='Gaussian'):
    joint_data = []
    marginal_data = []
   # loop_range=1
    for counter0 in range(loop_range):
      if mode=='auto':
        if inptype=='Gaussian':
          y,x=custom_Gaussian(128,snr)
        if inptype=='bpsk':
          y,x=custom_BPSK(128,snr)
        if inptype=='laplace':
          y,x=custom_laplace(128,snr)
        batch_size = tf.shape(x)[0]
        x_tiled = tf.tile(x[None, :],  (batch_size, 1, 1))
        y_tiled = tf.tile(y[:, None],  (1, batch_size, 1))
        xy_pairs = tf.concat((x_tiled, y_tiled), axis=2).numpy()

        mask = np.eye(xy_pairs.shape[0],dtype=bool)

        joint_buffer = xy_pairs[mask,:]
        marginal_buffer = xy_pairs[~mask,:]
        random_index = np.random.choice(batch_size * batch_size - batch_size, batch_size)
        marginal_buffer = marginal_buffer[random_index]


        joint_data.append(joint_buffer)
        marginal_data.append(marginal_buffer)
       #print(joint_data)
        joint_data1 = np.concatenate(joint_data, axis=0)
        marginal_data1 = np.concatenate(marginal_data, axis=0)
      else:

  
       batch_size = tf.shape(x)[0]
       x_tiled = tf.tile(x[None, :],  (batch_size, 1, 1))
       y_tiled = tf.tile(y[:, None],  (1, batch_size, 1))
       xy_pairs = tf.concat((x_tiled, y_tiled), axis=2).numpy()

       mask = np.eye(xy_pairs.shape[0],dtype=bool)

       joint_buffer = xy_pairs[mask,:]
       marginal_buffer = xy_pairs[~mask,:]
       random_index = np.random.choice(batch_size * batch_size - batch_size, batch_size)
       marginal_buffer = marginal_buffer[random_index]


       joint_data.append(joint_buffer)
       marginal_data.append(marginal_buffer)
       print(joint_data)
    joint_data1 = np.concatenate(joint_data, axis=0)
    marginal_data1 = np.concatenate(marginal_data, axis=0)
    
    return tf.convert_to_tensor(joint_data1, tf.float32), tf.convert_to_tensor(marginal_data1, tf.float32)