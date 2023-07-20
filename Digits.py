{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU",
    "gpuClass": "standard"
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "from matplotlib import pyplot as plt\n",
        "from keras.models import Sequential\n",
        "from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D\n"
      ],
      "metadata": {
        "id": "qREFVqW96fpx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from keras.utils import np_utils\n",
        "from keras.datasets import mnist"
      ],
      "metadata": {
        "id": "e0ieBiIG7M8Q"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "(Xtrain,Ytrain), (Xtest,Ytest)=mnist.load_data()"
      ],
      "metadata": {
        "id": "DTLoBquX7cMb",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "10e98ac3-16a7-4f08-926a-ef8c5cb0d7ed"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz\n",
            "11493376/11490434 [==============================] - 0s 0us/step\n",
            "11501568/11490434 [==============================] - 0s 0us/step\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.imshow(Xtrain[10])\n",
        "plt.show()\n",
        "print(Ytrain[10])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "id": "YGKsKtFC71RP",
        "outputId": "3fb3c2e9-b902-4e64-d6b2-e3f66d37d7e3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "3\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "Xtrain=Xtrain.reshape(Xtrain.shape[0],28,28,1)\n",
        "Xtest=Xtest.reshape(Xtest.shape[0],28,28,1)"
      ],
      "metadata": {
        "id": "2NIetogZ8A3Z"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "Xtrain=Xtrain.astype('float32')\n",
        "Xtest=Xtest.astype('float32')"
      ],
      "metadata": {
        "id": "4wQxHK0L8hkw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "Xtrain/=255\n",
        "Xtest/=255"
      ],
      "metadata": {
        "id": "ORGXYtNH8pmZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(Ytrain[10])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZxgMLrwP8wev",
        "outputId": "1c17fb4a-fdd9-461e-a2bd-2f3adf20a836"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "3\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "Ytrain=np_utils.to_categorical(Ytrain,10)\n",
        "Ytest=np_utils.to_categorical(Ytest,10)"
      ],
      "metadata": {
        "id": "-WXTavZE80JK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(Ytrain[10])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XazO7lJ89B9a",
        "outputId": "a692274d-3358-4035-e68d-c6ec8f931067"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model=Sequential()\n",
        "model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(28,28,1)))\n",
        "model.add(MaxPooling2D(pool_size=(2,2)))\n",
        "model.add(Convolution2D(32,(3,3),activation='relu'))\n",
        "model.add(MaxPooling2D(pool_size=(2,2)))\n",
        "\n",
        "model.add(Flatten())\n",
        "model.add(Dense(64,activation='relu'))\n",
        "model.add(Dense(10,activation='softmax'))"
      ],
      "metadata": {
        "id": "53nQ-IYP9GR5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.summary()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "28S_Un6W-Hgj",
        "outputId": "2ba3b01d-6777-41c1-be5b-d02ba09acb94"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model: \"sequential\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " conv2d (Conv2D)             (None, 26, 26, 32)        320       \n",
            "                                                                 \n",
            " max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0         \n",
            " )                                                               \n",
            "                                                                 \n",
            " conv2d_1 (Conv2D)           (None, 11, 11, 32)        9248      \n",
            "                                                                 \n",
            " max_pooling2d_1 (MaxPooling  (None, 5, 5, 32)         0         \n",
            " 2D)                                                             \n",
            "                                                                 \n",
            " flatten (Flatten)           (None, 800)               0         \n",
            "                                                                 \n",
            " dense (Dense)               (None, 64)                51264     \n",
            "                                                                 \n",
            " dense_1 (Dense)             (None, 10)                650       \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 61,482\n",
            "Trainable params: 61,482\n",
            "Non-trainable params: 0\n",
            "_________________________________________________________________\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])"
      ],
      "metadata": {
        "id": "iHVrRV0J-K62"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.fit(Xtrain,Ytrain,batch_size=1024,epochs=10,verbose=1)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EBCd43ro-eEy",
        "outputId": "d8b67da7-4696-4c60-d034-cbaeebdf8ab5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "59/59 [==============================] - 13s 18ms/step - loss: 0.9340 - accuracy: 0.7461\n",
            "Epoch 2/10\n",
            "59/59 [==============================] - 1s 16ms/step - loss: 0.1907 - accuracy: 0.9434\n",
            "Epoch 3/10\n",
            "59/59 [==============================] - 1s 16ms/step - loss: 0.1183 - accuracy: 0.9648\n",
            "Epoch 4/10\n",
            "59/59 [==============================] - 1s 16ms/step - loss: 0.0896 - accuracy: 0.9732\n",
            "Epoch 5/10\n",
            "59/59 [==============================] - 1s 16ms/step - loss: 0.0743 - accuracy: 0.9776\n",
            "Epoch 6/10\n",
            "59/59 [==============================] - 1s 16ms/step - loss: 0.0633 - accuracy: 0.9816\n",
            "Epoch 7/10\n",
            "59/59 [==============================] - 1s 16ms/step - loss: 0.0570 - accuracy: 0.9829\n",
            "Epoch 8/10\n",
            "59/59 [==============================] - 1s 16ms/step - loss: 0.0515 - accuracy: 0.9844\n",
            "Epoch 9/10\n",
            "59/59 [==============================] - 1s 16ms/step - loss: 0.0462 - accuracy: 0.9861\n",
            "Epoch 10/10\n",
            "59/59 [==============================] - 1s 16ms/step - loss: 0.0420 - accuracy: 0.9873\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7f97b00fdc10>"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!nvidia-smi"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JUhqSm1z-oZT",
        "outputId": "9c31618e-b932-48d5-8136-4edcea54bb71"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Wed Aug 24 08:24:38 2022       \n",
            "+-----------------------------------------------------------------------------+\n",
            "| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |\n",
            "|-------------------------------+----------------------+----------------------+\n",
            "| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |\n",
            "| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |\n",
            "|                               |                      |               MIG M. |\n",
            "|===============================+======================+======================|\n",
            "|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |\n",
            "| N/A   64C    P0    39W /  70W |   1836MiB / 15109MiB |     73%      Default |\n",
            "|                               |                      |                  N/A |\n",
            "+-------------------------------+----------------------+----------------------+\n",
            "                                                                               \n",
            "+-----------------------------------------------------------------------------+\n",
            "| Processes:                                                                  |\n",
            "|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |\n",
            "|        ID   ID                                                   Usage      |\n",
            "|=============================================================================|\n",
            "+-----------------------------------------------------------------------------+\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pred=model.predict(Xtest[np.newaxis,1])"
      ],
      "metadata": {
        "id": "o00ad2AH_eia"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pred"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "N3XHCWcV_udZ",
        "outputId": "57443fe9-a23f-48b6-8a4a-6b92a740baa4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[1.3581294e-05, 4.1940325e-04, 9.9955136e-01, 6.1087562e-06,\n",
              "        4.9651277e-12, 1.7615594e-08, 4.0506852e-06, 8.8870511e-10,\n",
              "        5.4345251e-06, 6.5010612e-14]], dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 18
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "Predication=pred.argmax(axis=1)"
      ],
      "metadata": {
        "id": "De37xt3i_vOJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "Predication"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zhjuer4u_0TS",
        "outputId": "53341d55-63a7-4da9-ca5e-888bca6ea1e9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([2])"
            ]
          },
          "metadata": {},
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(Predication[0])\n",
        "img=(Xtest[1]*255).reshape((28,28)).astype('uint8')\n",
        "plt.imshow(img)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 282
        },
        "id": "ccDG8Yei_3Kb",
        "outputId": "a461d76a-e527-4137-831a-177107ed894b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "for i in np.random.choice(np.arange(0,len(Ytest)),size=(10,)):\n",
        "  pred=model.predict(Xtest[np.newaxis,i])\n",
        "  prediction=pred.argmax(axis=1)\n",
        "  img=(Xtest[i]*255).reshape((28,28)).astype('uint8')\n",
        "  print(prediction[0])\n",
        "  plt.imshow(img)\n",
        "  plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "6TAlYObPATit",
        "outputId": "1cb3e9b1-e2c8-48c9-b317-7e4fad0326a9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "9\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "5\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "5\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "6\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPsAAAD4CAYAAAAq5pAIAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAANy0lEQVR4nO3df6zV9X3H8ddLuGJFTaG4OwZErWVbWZdCvaHWH4vO1FKyDGsWUro0NHO7bhPTLi6b6ZrVJW4jXX+sM1s3LCguFmNiKSxxW5GZmGaWeTUUAUWtRYUhDFmK3Trkx3t/3K/mgvd8z+V8v+d8D76fj+TmnPt9n+/5vD3eF99zvp9zzscRIQDvfGc13QCA3iDsQBKEHUiCsANJEHYgicm9HOxsT4lzNLWXQwKp/J/+R2/EEY9XqxR224skfV3SJEnfjIiVZbc/R1P1YV9XZUgAJbbE5pa1jp/G254k6W8lfVzSPEnLbM/r9P4AdFeV1+wLJb0QES9GxBuSHpC0pJ62ANStSthnSXplzO97im0nsT1se8T2yFEdqTAcgCq6fjY+IlZFxFBEDA1oSreHA9BClbDvlTRnzO+zi20A+lCVsD8haa7tS2yfLemTkjbW0xaAunU89RYRx2yvkPSvGp16WxMRO2rrDECtKs2zR8TDkh6uqRcAXcTbZYEkCDuQBGEHkiDsQBKEHUiCsANJ9PTz7DjznLh6QWn9xZvL93/22m92PPavzbqs433xdhzZgSQIO5AEYQeSIOxAEoQdSIKwA0kw9YZK7rtidWn9hE50fN8//PLlpfVL//D7Hd93RhzZgSQIO5AEYQeSIOxAEoQdSIKwA0kQdiAJ5tmTmzxndmn9o3//aGl9aMrxOttBF3FkB5Ig7EAShB1IgrADSRB2IAnCDiRB2IEkmGd/p7NLyzv/7GdL699594bSeuefVm+Pz6vXq1LYbe+W9Lqk45KORcRQHU0BqF8dR/ZrI+JgDfcDoIt4zQ4kUTXsIem7tp+0PTzeDWwP2x6xPXJURyoOB6BTVZ/GXxURe23/jKRNtp+NiMfG3iAiVklaJUkXeHpUHA9Ahyod2SNib3F5QNJ6SQvraApA/ToOu+2pts9/87qk6yVtr6sxAPWq8jR+UNJ6j87jTpb0rYj4l1q6Qm3+87aPlNaf/djX29xDtdM69/z44pa19fMurHTfOD0dhz0iXpT0wRp7AdBFTL0BSRB2IAnCDiRB2IEkCDuQBB9xfQf46Q2t38u08dYvtdl7Sr3NnOJbr7Tu7V36UVfHxsk4sgNJEHYgCcIOJEHYgSQIO5AEYQeSIOxAEsyz94M2X/fc7mOqiz71eMvaz02uNo/+zNGjpfUVu5aV1s+/6Y2WtWMddYROcWQHkiDsQBKEHUiCsANJEHYgCcIOJEHYgSSYZ+8Dr/325aX1kT9o93XP3bPk31aU1n/+t0ZK68yl9w+O7EAShB1IgrADSRB2IAnCDiRB2IEkCDuQBPPsfeC1K1p/5luSzmrw3+T3PD7Q2NioV9u/IttrbB+wvX3Mtum2N9l+vric1t02AVQ1kUPGvZIWnbLtdkmbI2KupM3F7wD6WNuwR8Rjkg6dsnmJpLXF9bWSbqi5LwA16/Q1+2BE7CuuvyppsNUNbQ9LGpakc3Ruh8MBqKrymZ+ICElRUl8VEUMRMTTQ5UUEAbTWadj3254pScXlgfpaAtANnYZ9o6TlxfXlkjbU0w6Abmn7mt32OknXSJphe4+kL0paKelB2zdJeknS0m42eaY7/Knyz6tvuPavS+snNKnOdk4y78FbS+vvu7v1d9LjzNI27BHRahWA62ruBUAX8XZZIAnCDiRB2IEkCDuQBGEHkuAjrj0w7XdeLq3/wkD3ptaq8oJfqrT/rt9r/Rbp98/dW7rvWW75xkxJ0okoX+r64L0XtaxN33a4dN9JB39cWj/2yp7Sej/iyA4kQdiBJAg7kARhB5Ig7EAShB1IgrADSTDPntzOpXeV1s9aWn48OKETdbZz8thtjkVtx/6Lzsf+q9d+ubS+bt2vltZn/+W/dz54l3BkB5Ig7EAShB1IgrADSRB2IAnCDiRB2IEkmGfvgXafy25ySeZ2Blz+Wfuj5f9pZ+zYX5jxbGn93N8sX2b7kQ1DpfXjO5877Z6q6t+/MgC1IuxAEoQdSIKwA0kQdiAJwg4kQdiBJJhnr8HkObNL6++/oPz70bv5mfCq2s1ld7P3fh77lmm7Suu77h0sre9eeLodVdf2yG57je0DtreP2XaH7b22txY/i7vbJoCqJvI0/l5Ji8bZ/rWImF/8PFxvWwDq1jbsEfGYpEM96AVAF1U5QbfC9rbiaf60VjeyPWx7xPbIUR2pMByAKjoN+zckXSppvqR9kr7S6oYRsSoihiJiaEBTOhwOQFUdhT0i9kfE8Yg4IeluSQ2cWwRwOjoKu+2ZY379hKTtrW4LoD+0nWe3vU7SNZJm2N4j6YuSrrE9X1JI2i3p5i722PcOXV0+z37n4PoeddJ/Hvnp+S1rf/6Fz5Tu22b5dbX5mgBd+UdbWtbuHPyP8p0rWnDey6X13bqwq+OPp23YI2LZOJtXd6EXAF3E22WBJAg7kARhB5Ig7EAShB1Igo+41uC1X//fplvoW7f+8/KWtbkPfL903+fuuaz8ztvMzV02dXf5/hX8zX//Ymn9/n/4WGl9UL1f0pkjO5AEYQeSIOxAEoQdSIKwA0kQdiAJwg4kwTz7BL38p1e0rO24+q42e5+5/6ZWXTZ5141/17p4Y7uxt7YZ+3j5HZQq/3/S7r/77n+6vrR+yV29n0dv58z9KwRwWgg7kARhB5Ig7EAShB1IgrADSRB2IAnm2Seo7GuL+3nJ5ar6ednkKmO3+zx6u3n0997xVGm9TeuN4MgOJEHYgSQIO5AEYQeSIOxAEoQdSIKwA0kwzz5BFz90sGVt5W98sHTf22f8oO52Urjv8KzS+tqXP1JaP3bPYMvau7cdKt33kp2Pl9b7cR69nbZHdttzbD9qe6ftHbY/W2yfbnuT7eeLy2ndbxdApybyNP6YpNsiYp6kyyXdYnuepNslbY6IuZI2F78D6FNtwx4R+yLiqeL665KekTRL0hJJa4ubrZV0Q7eaBFDdab1mt32xpAWStkgajIh9RelVSeO+QLI9LGlYks7RuZ32CaCiCZ+Nt32epIckfS4iDo+tRUSoxTmLiFgVEUMRMTSgKZWaBdC5CYXd9oBGg35/RHy72Lzf9syiPlPSge60CKAObZ/G27ak1ZKeiYivjiltlLRc0srickNXOuwTx3c+17K2ZdkHSvddcOM1pfUPLd5ZWl990abSej/bcmSgZe131/x+6b5z7iz/OuZ36UdtRm9dr/Il1Geqibxmv1LSpyU9bb/1Rd6f12jIH7R9k6SXJC3tTosA6tA27BHxPUmtVr2/rt52AHQLb5cFkiDsQBKEHUiCsANJEHYgCY+++a03LvD0+LA5gQ90y5bYrMNxaNzZM47sQBKEHUiCsANJEHYgCcIOJEHYgSQIO5AEYQeSIOxAEoQdSIKwA0kQdiAJwg4kQdiBJAg7kARhB5Ig7EAShB1IgrADSRB2IAnCDiRB2IEkCDuQRNuw255j+1HbO23vsP3ZYvsdtvfa3lr8LO5+uwA6NZH12Y9Jui0inrJ9vqQnbW8qal+LiC93rz0AdZnI+uz7JO0rrr9u+xlJs7rdGIB6ndZrdtsXS1ogaUuxaYXtbbbX2J7WYp9h2yO2R47qSKVmAXRuwmG3fZ6khyR9LiIOS/qGpEslzdfokf8r4+0XEasiYigihgY0pYaWAXRiQmG3PaDRoN8fEd+WpIjYHxHHI+KEpLslLexemwCqmsjZeEtaLemZiPjqmO0zx9zsE5K2198egLpM5Gz8lZI+Lelp21uLbZ+XtMz2fEkhabekm7vSIYBaTORs/Pckjbfe88P1twOgW3gHHZAEYQeSIOxAEoQdSIKwA0kQdiAJwg4kQdiBJAg7kARhB5Ig7EAShB1IgrADSRB2IAlHRO8Gs/9L0ktjNs2QdLBnDZyefu2tX/uS6K1TdfZ2UURcOF6hp2F/2+D2SEQMNdZAiX7trV/7kuitU73qjafxQBKEHUii6bCvanj8Mv3aW7/2JdFbp3rSW6Ov2QH0TtNHdgA9QtiBJBoJu+1FtnfZfsH27U300Irt3bafLpahHmm4lzW2D9jePmbbdNubbD9fXI67xl5DvfXFMt4ly4w3+tg1vfx5z1+z254k6TlJH5W0R9ITkpZFxM6eNtKC7d2ShiKi8Tdg2P4VST+RdF9EfKDY9iVJhyJiZfEP5bSI+OM+6e0OST9pehnvYrWimWOXGZd0g6TPqMHHrqSvperB49bEkX2hpBci4sWIeEPSA5KWNNBH34uIxyQdOmXzEklri+trNfrH0nMteusLEbEvIp4qrr8u6c1lxht97Er66okmwj5L0itjft+j/lrvPSR91/aTtoebbmYcgxGxr7j+qqTBJpsZR9tlvHvplGXG++ax62T586o4Qfd2V0XEhyR9XNItxdPVvhSjr8H6ae50Qst498o4y4y/pcnHrtPlz6tqIux7Jc0Z8/vsYltfiIi9xeUBSevVf0tR739zBd3i8kDD/byln5bxHm+ZcfXBY9fk8udNhP0JSXNtX2L7bEmflLSxgT7exvbU4sSJbE+VdL36bynqjZKWF9eXS9rQYC8n6ZdlvFstM66GH7vGlz+PiJ7/SFqs0TPyP5T0J0300KKv90r6QfGzo+neJK3T6NO6oxo9t3GTpPdI2izpeUmPSJreR739o6SnJW3TaLBmNtTbVRp9ir5N0tbiZ3HTj11JXz153Hi7LJAEJ+iAJAg7kARhB5Ig7EAShB1IgrADSRB2IIn/B0O5DnpN/S+gAAAAAElFTkSuQmCC\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "6\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "5\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "6\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "37T9qOOrBXA0"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}