    # Build Discriminator
    self.discriminator = build_discriminator(
        img_shape=self.img_shape,
        num_classes=self.num_classes
    )
    self.discriminator.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    # Build Generator
    self.generator = build_generator(
        z_dimension=self.z_dimension,
        img_shape=self.img_shape,
        num_classes=self.num_classes
    )
    noise = Input(shape=(self.z_dimension,)) # Input for Generator
    img = self.generator([noise]) # Generator generates an image
    self.discriminator.trainable = False # Set the discriminator in non-trainable mode
    d_pred = self.discriminator([img]) # Generator image as input for the discriminator
    self.combined = Model(
        inputs=[noise],
        outputs=d_pred
    )
    self.combined.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=[]
    )