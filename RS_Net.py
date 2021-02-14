
def RS_Net(shape, classes=1):
    inputs = Input(shape) # [512, 512, 3]
    pool1 = BatchNormalization()(inputs)

    def saliency_block(x0, filter_num):
        x1=CONV2D(x0, filter_num, (1, 1))
        x1=CONV2D(x1, 1, (1, 1), activation='sigmoid')
        x1=CONV2D(x1, filter_num, (1, 1))
        x1 = Multiply()([x0, x1])
        x1=Add()([x0, x1])
        return x1
    
    global conv1, conv2, conv3, conv4, conv5, conv6
    global conv1a, conv2a, conv3a, conv4a, conv5a
    conv1, conv2, conv3, conv4, conv5, conv6 = None, None, None, None, None, None
    conv1a, conv2a, conv3a, conv4a, conv5a = None, None, None, None, None

    if conv1 is not None: 
        conv1 = saliency_block(conv1, 32)
        pool1 = Concatenate()([pool1, conv1]); 
    if conv1a is not None: 
        conv1a = saliency_block(conv1a, 32)
        pool1 = Concatenate()([pool1, conv1a]); 
    conv0 = CONV2D(pool1, 32, (3, 3));    conv1 = CONV2D(conv0, 32, (3, 3));
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1);  # 512/2

    if conv2 is not None: 
        conv2 = saliency_block(conv2, 64)
        pool1 = Concatenate()([pool1, conv2]); 
    if conv2a is not None: 
        conv2a = saliency_block(conv2a, 64)
        pool1 = Concatenate()([pool1, conv2a]); 
    conv0 = CONV2D(pool1, 64, (3, 3));    conv2 = CONV2D(conv0, 64, (3, 3));
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2);  # 512/4

    if conv3 is not None: 
        conv3 = saliency_block(conv3, 128)
        pool1 = Concatenate()([pool1, conv3]); 
    if conv3a is not None: 
        conv3a = saliency_block(conv3a, 128)
        pool1 = Concatenate()([pool1, conv3a]); 
    conv0 = CONV2D(pool1, 128, (3, 3));    conv3 = CONV2D(conv0, 128, (3, 3));
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv3);  # 512/8

    if conv4 is not None: 
        conv4 = saliency_block(conv4, 256)
        pool1 = Concatenate()([pool1, conv4]); 
    if conv4a is not None: 
        conv4a = saliency_block(conv4a, 256)
        pool1 = Concatenate()([pool1, conv4a]); 
    conv0 = CONV2D(pool1, 256, (3, 3));    conv4 = CONV2D(conv0, 256, (3, 3));
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv4);  # 512/16

    if conv5 is not None: 
        conv5 = saliency_block(conv5, 512)
        pool1 = Concatenate()([pool1, conv5]); 
    if conv5a is not None: 
        conv5a = saliency_block(conv5a, 512)
        pool1 = Concatenate()([pool1, conv5a]); 
    conv0 = CONV2D(pool1, 512, (3, 3));    conv5 = CONV2D(conv0, 512, (3, 3));
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv5);  # 512/32

    #----------------------------------------------
    if conv6 is not None: 
        conv6 = saliency_block(conv6, 1024)
        pool1 = Concatenate()([pool1, conv6]); 
    conv0 = CONV2D(pool1, 1024, (3, 3));    conv6 = CONV2D(conv0, 1024, (3, 3)); # 512/32
    #----------------------------------------------

    merg1 = UpSampling2D(size=(2, 2))(conv6);
    if conv5a is not None: 
        conv5a = saliency_block(conv5a, 512)
        merg1 = Concatenate()([merg1, conv5a]); 
    conv0 = saliency_block(conv5, 512)
    merg1 = Concatenate()([merg1, conv0]) # 512/16
    conv0 = CONV2D(merg1, 512, (3, 3));    conv5a = CONV2D(conv0, 512, (3, 3));
    
    merg1 = UpSampling2D(size=(2, 2))(conv5a);
    if conv4a is not None: 
        conv4a = saliency_block(conv4a, 256)
        merg1 = Concatenate()([merg1, conv4a]); 
    conv0 = saliency_block(conv4, 256)
    merg1 = Concatenate()([merg1, conv0]) # 512/8
    conv0 = CONV2D(merg1, 256, (3, 3));    conv4a = CONV2D(conv0, 256, (3, 3));

    merg1 = UpSampling2D(size=(2, 2))(conv4a);
    if conv3a is not None: 
        conv3a = saliency_block(conv3a, 128)
        merg1 = Concatenate()([merg1, conv3a]); 
    conv0 = saliency_block(conv3, 128)
    merg1 = Concatenate()([merg1, conv0]) # 512/4
    conv0 = CONV2D(merg1, 128, (3, 3));    conv3a = CONV2D(conv0, 128, (3, 3));

    merg1 = UpSampling2D(size=(2, 2))(conv3a);
    if conv2a is not None: 
        conv2a = saliency_block(conv2a, 64)
        merg1 = Concatenate()([merg1, conv2a]); 
    conv0 = saliency_block(conv2, 64)
    merg1 = Concatenate()([merg1, conv0]) # 512/4
    conv0 = CONV2D(merg1, 64, (3, 3));    conv2a = CONV2D(conv0, 64, (3, 3));

    merg1 = UpSampling2D(size=(2, 2))(conv2a);
    if conv1a is not None: 
        conv1a = saliency_block(conv1a, 32)
        merg1 = Concatenate()([merg1, conv1a]); 
    conv0 = saliency_block(conv1, 32)
    merg1 = Concatenate()([merg1, conv0]) # 512/2
    conv0 = CONV2D(merg1, 32, (3, 3));    conv1a = CONV2D(conv0, 32, (3, 3));

    conv0 = CONV2D(conv1a, classes, (1, 1), activation='sigmoid')
    model = Model(input=inputs, output=conv0)
    model.summary() 
    return model

