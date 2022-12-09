

@tf.function
def train_step(joint_data, marginal_data):
    with tf.GradientTape() as tape:
        pure_loss_value,t__ = mmse_loss(joint_data, marginal_data)

        loss_value = pure_loss_value + sum(mmse_model.losses)
        
    grads = tape.gradient(loss_value, mmse_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, mmse_model.trainable_weights))

    loss_tracker.update_state(pure_loss_value)
    

@tf.function
def validation_step(joint_data, marginal_data):
    loss_value,t__ = mmse_loss(joint_data, marginal_data)
    
    loss_tracker.update_state(loss_value)
def train_MMSE(batch_size,snr,inptype='Gaussian'):
    epochs = 3
    batch_size = batch_size
    nsample_train = 100 * batch_size
    nsample_test = 50 * batch_size



    #train_x, train_noise = x_distribution.sample(nsample_train), noise_distribution.sample(nsample_train)
    if inptype=='Gaussian':
      train_x,train_y=custom_Gaussian(nsample_train,snr)
    if inptype=='bpsk':
      train_x,train_y=custom_BPSK(nsample_train,snr)
    if inptype=='laplace':
      train_x,train_y=custom_laplace(nsample_train,snr)


    train_dataset_x = tf.data.Dataset.from_tensor_slices((train_x))
    train_dataset_y = tf.data.Dataset.from_tensor_slices((train_y))
    train_dataset_x = train_dataset_x.shuffle(10 ** 4).batch(batch_size)
    train_dataset_y = train_dataset_y.shuffle(10 ** 4).batch(batch_size)
    train_dataset = tf.data.Dataset.zip((train_dataset_x, train_dataset_y)).prefetch(tf.data.AUTOTUNE)


    #test_x, test_noise = x_distribution.sample(nsample_test),noise_distribution.sample(nsample_test)
    if inptype=='Gaussian':
      test_x,test_y=custom_Gaussian(nsample_test,snr)
    if inptype=='bpsk':
      test_x,test_y=custom_BPSK(nsample_test,snr)
    if inptype=='laplace':
      test_x,test_y=custom_laplace(nsample_test,snr)
   # test_x,test_y=custom_Gaussian(nsample_test,10)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)



    all_train_loss = []
    all_validation_loss = []

    for epoch in range(epochs):
        print(f"Epoch : {epoch + 1} / {epochs}")
        
        pbar = tqdm(train_dataset, total=nsample_train // batch_size)
        for x, y in pbar:
            
            joint_data, marginal_data = joint_marginal_batch(x, y)
            print(marginal_data.shape)
            train_step(joint_data, marginal_data)
            
            pbar.set_description("Epoch %4d / %4d : Train Loss : ( %.4f )" % \
                                (epoch + 1, epochs, float(loss_tracker.result())))
        
            all_train_loss.append(float(loss_tracker.result()))
        loss_tracker.reset_states()
        
        
        pbar = tqdm(test_dataset, total=nsample_test // batch_size)
        for x, y in pbar:
            
            joint_data, marginal_data = joint_marginal_batch(x, y)
            
            validation_step(joint_data, marginal_data)
            
            pbar.set_description("Epoch %4d / %4d : Valid Loss : ( %.4f ) " % \
                                (epoch + 1, epochs, float(loss_tracker.result())))
        
            all_validation_loss.append(float(loss_tracker.result()))
        loss_tracker.reset_states()
    return all_validation_loss,all_train_loss