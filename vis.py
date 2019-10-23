import matplotlib.pyplot as plt

def show_digits(images, w, h):
  for i, image in enumerate(images):
    plt.subplot(w, h, i+1)
    plt.imshow(image.reshape(28,28), cmap='gray')
    plt.axis('off')