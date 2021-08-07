from main import base_model

model = load_model("mask_detector_1.model")

base_model.trainable = True

model.summary()
opt = tf.keras.optimizers.Adam(1e-5)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, epochs=10, validation_data=val_data)

