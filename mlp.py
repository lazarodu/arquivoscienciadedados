import numpy as np
import random

class mlp:

  def __init__(self, input_size, num_hidden, output_size, learn, momentum = 0):
    # cada neurônio oculto tem um peso por entrada, mais um peso bias
    self.hidden_layer = [[random.random() for _ in range(input_size + 1)]
                    for _ in range(num_hidden)]
    # cada neurônio de saída tem um peso por neurônio oculto, mais o peso bias
    self.output_layer = [[random.random() for _ in range(num_hidden + 1)]
                    for _ in range(output_size)]
    # a rede começa com pesos aleatórios
    self.network = [self.hidden_layer, self.output_layer]
    self.learn = learn
    self.momentum = momentum
    self.em = list()

  def sigmoid(self, t):
    return 1 / (1 + np.exp(-t))

  def neuron_output(self, weights, inputs):
    return self.sigmoid(np.dot(weights, inputs))

  def feed_forward(self, input_vector):
    #recebe a rede neural
    #(representada como uma lista de listas de listas de pesos)
    #e retorna a saída a partir da entrada a se propagar
    outputs = []
    #processa uma camada por vez
    for layer in self.network:
      input_with_bias = input_vector + [1]              # adiciona uma entrada polarizada
      output = [self.neuron_output(neuron, input_with_bias)  # computa a saída
                for neuron in layer]                    # para cada neurônio
      outputs.append(output)                            # e memoriza
      # então a entrada para a próxima camada é a saída desta
      input_vector = output

    return outputs

  def backpropagate(self, input_vector, targets):
    hidden_outputs, outputs = self.feed_forward(input_vector)
    #a saída * (1 - output) é da derivada da sigmoid
    output_deltas = [output * (1 - output) * (output - target)
                    for output, target in zip(outputs, targets)]
    #ajusta os pesos para a camada de saída, um neurônio por vez
    for i, output_neuron in enumerate(self.network[-1]):
      #foca no i-ésimo neurônio da camada de saída
      for j, hidden_output in enumerate(hidden_outputs + [1]):
        # ajusta o j-ésimo peso baseado em ambos
        # o delta deste neurônio e sua j-ésima entrada
        if self.momentum == 0:
          output_neuron[j] -= output_deltas[i] * hidden_output * self.learn
        else:
          output_neuron[j] -= output_deltas[i] * hidden_output * self.learn + self.momentum * output_deltas[i] * hidden_output
    #erros de backpropagation para a camada oculta
    hidden_deltas = [hidden_output * (1 - hidden_output) * 
                    np.dot(output_deltas, [n[i] for n in self.output_layer])
                    for i, hidden_output in enumerate(hidden_outputs)]
    #ajusta os pesos para a camada oculta, um neurônio por vez
    for i, hidden_neuron in enumerate(self.network[0]):
      for j, input in enumerate(input_vector + [1]):
        if self.momentum == 0:
          hidden_neuron[j] -= hidden_deltas[i] * input * self.learn
        else:
          hidden_neuron[j] -= hidden_deltas[i] * input * self.learn + self.momentum * hidden_deltas[i] * input

  def predict(self, input):
    return self.feed_forward(input)

  def save_network(self):
    for i in range(len(self.network)):
      np.savetxt('w'+str(i)+'.csv', (self.network[i]), fmt='%1.4e', delimiter=',', newline='\n')

  def load_network(self, num_weights):
    self.network = list()
    for i in range(num_weights):
      self.network.append(np.genfromtxt('w'+str(i)+'.csv',dtype='float',delimiter=','))

  def mean_square_error(self, y, y_pred):
    self.em.append(np.mean(((y) - (y_pred))**2))

  def train(self, epochs, x, y, precisao):
    epocas = 0
    for _ in range(epochs):
      for input_vector, target_vector in zip(x,y): #zip(inputs, targets):
        self.backpropagate(input_vector, target_vector)
      # avalia
      pred = list()
      for i in range(len(x)):
        pred.append(self.predict(x[i])[1])
      self.mean_square_error(np.array(y),np.array(pred))
      if epocas > 0:
        if np.absolute(self.em[epocas-1] - self.em[epocas]) < precisao:
          break
      epocas += 1
    print(epocas)
