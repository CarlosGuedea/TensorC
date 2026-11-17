#!/usr/bin/env python3
"""Same normalized Fahrenheit -> Celsius training loop as examples/fahrenheit.c."""

import sys
import time


SAMPLE_COUNT = 2049
EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 0.20
FAHRENHEIT_MIN = -40.0
FAHRENHEIT_MAX = 212.0
CELSIUS_MIN = -40.0
CELSIUS_MAX = 100.0


def normalize_fahrenheit(value):
    return (value - FAHRENHEIT_MIN) / (FAHRENHEIT_MAX - FAHRENHEIT_MIN)


def denormalize_celsius(value):
    return value * (CELSIUS_MAX - CELSIUS_MIN) + CELSIUS_MIN


def main():
    x = []
    y = []
    for i in range(SAMPLE_COUNT):
        fahrenheit = FAHRENHEIT_MIN + (FAHRENHEIT_MAX - FAHRENHEIT_MIN) * i / (SAMPLE_COUNT - 1)
        celsius = (fahrenheit - 32.0) * (5.0 / 9.0)
        x.append(normalize_fahrenheit(fahrenheit))
        y.append((celsius - CELSIUS_MIN) / (CELSIUS_MAX - CELSIUS_MIN))

    weight = 0.0
    bias = 0.0
    start = time.perf_counter()
    loss = 0.0
    for _ in range(EPOCHS):
        total_loss = 0.0
        for start_index in range(0, SAMPLE_COUNT, BATCH_SIZE):
            end_index = min(start_index + BATCH_SIZE, SAMPLE_COUNT)
            gradient_weight = 0.0
            gradient_bias = 0.0
            for input_value, target in zip(x[start_index:end_index], y[start_index:end_index]):
                prediction = input_value * weight + bias
                error = prediction - target
                total_loss += error * error
                # Matches TensorC 0.2's dense-layer SGD gradient convention.
                gradient_weight += input_value * error
                gradient_bias += error
            count = end_index - start_index
            weight -= LEARNING_RATE * gradient_weight / count
            bias -= LEARNING_RATE * gradient_bias / count
        loss = total_loss / SAMPLE_COUNT
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    sample = normalize_fahrenheit(100.0)
    prediction = denormalize_celsius(sample * weight + bias)
    print("language=Python")
    print(f"samples={SAMPLE_COUNT} epochs={EPOCHS} batch_size={BATCH_SIZE}")
    print(f"elapsed_ms={elapsed_ms:.3f}")
    print(f"final_mse={loss:.8f}")
    print(f"100F_to_C={prediction:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
