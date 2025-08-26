from django.test import TestCase
from ml.predict import predict
pred, probs = predict("10.png")  # use any 28x28 grayscale digit image
print("Predicted:", pred)
print("Probabilities:", probs)


# Create your tests here.
