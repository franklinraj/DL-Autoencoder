# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.
## DESIGN STEPS
### STEP 1: 
Define the Denoising Autoencoder model with convolution layers for encoding and transposed convolution layers for decoding the image.



### STEP 2: 
Initialize the model, loss function (MSELoss), and optimizer (Adam) for training.




### STEP 3: 

Load images from the dataset and add noise to the images to create noisy inputs.



### STEP 4: 
Pass the noisy images through the autoencoder to reconstruct clean images.




### STEP 5: 

Compute the loss between reconstructed and original images, then perform backpropagation and update model weights.



### STEP 6: 
Repeat the process for several epochs and visualize the denoised output images after training.





## PROGRAM

### Name:Franklin raj g

### Register Number:212223230058

```python
# Autoencoder Definition
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 32, 7, 7]
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # [B, 1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training function
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    print("Name:franklin raj g                   ")
    print("Register Number:   212223230058               ")

    for epoch in range(epochs):
        running_loss = 0.0

        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            # Forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")
# Visualization function
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)

```

### OUTPUT

### Model Summary
<img width="1029" height="546" alt="image" src="https://github.com/user-attachments/assets/e2e34ac6-9be8-4c55-957b-2568ffe93d9b" />


### Training loss
<img width="333" height="219" alt="image" src="https://github.com/user-attachments/assets/e0511550-2902-497d-8491-428870d73025" />

## Original vs Noisy Vs Reconstructed Image
<img width="1750" height="583" alt="image" src="https://github.com/user-attachments/assets/7724df31-8101-41a2-900c-5bd79cd15a70" />


## RESULT
Thus,To develop a convolutional autoencoder for image denoising application,has been done by using pytorch.
