from flask import Flask, request, render_template, jsonify
# from flask_cors import CORS
app = Flask(__name__)
# CORS(app) #If need be

import scipy.io
import pandas as pd
import numpy as np
import random
import copy
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#Loading Data
def LoadData():

  dataset = scipy.io.loadmat('data1.mat')

  data = {}
  data['x'] = dataset['Inputs']
  data['t'] = dataset['Targets']

  data['nx'] = np.shape(data['x'])[0]
  data['nt'] = np.shape(data['t'])[0]
  data['nSample'] = np.shape(data['x'])[1]

  return data


def CreateRandomSolution(data=None):
  nx = data['nx']
  sol = np.random.permutation(range(nx))
  return sol


def RouletteWheelSelection(p=None):
  random_number = np.random.uniform()
  
  cum_list = []
  cum_sum = 0
  for i in range(0, len(p)):
    cum_sum += p[0][i]
    cum_list.append(cum_sum)
  
  for i in range(0, len(cum_list)):
    if random_number <= cum_list[i]:
      return i+1
  return len(cum_list)

def ApplySwap(tour1=None):
  length = len(tour1)
  #Selecting 2 number from 1-to-length of tour
  random_sample = random.sample(range(length), 2)

  first = random_sample[0]
  second = random_sample[1]

  tour2 = copy.deepcopy(tour1)
  tour2[first] = tour1[second]
  tour2[second] = tour1[first]

  return tour2

def ApplyReversion(tour1=None):
  length = len(tour1)
  #Selecting 2 number from 1-to-length of tour
  random_sample = random.sample(range(length), 2)

  first = min(random_sample)
  second = max(random_sample)

  tour2 = []
  for i in range(0, first):
    tour2.append(tour1[i])
  for i in range(second, first-1, -1):
    tour2.append(tour1[i])
  for i in range(second+1, length):
    tour2.append(tour1[i])

  return tour2


def ApplyInsertion(tour1=None):
  length = len(tour1)
  #Selecting 2 number from 1-to-length of tour
  random_sample = random.sample(range(length), 2)

  first = random_sample[0]
  second = random_sample[1]

  tour2 = []
  if first<second:
    for i in range(0, first):
      tour2.append(tour1[i])
    for i in range(first+1, second+1):
      tour2.append(tour1[i])
    tour2.append(tour1[first])
    for i in range(second+1, length):
      tour2.append(tour1[i])
  else:
    for i in range(0, second+1):
      tour2.append(tour1[i])
    tour2.append(tour1[first])
    for i in range(second+1, first):
      tour2.append(tour1[i])
    for i in range(first+1, length):
      tour2.append(tour1[i])
  
  return tour2


def CreateNeighbor(tour1=None):
  pswap = 0.2
  pReversion = 0.5
  pInsertion = 1 - pSwap - pReversion

  p = []
  p.append(pswap)
  p.append(pReversion)
  p.append(pInsertion)

  method = RouletteWheelSelection(p)
  if method == 1:
    tour2 = ApplySwap(tour1)
  elif method == 2:
    tour2 = ApplyReversion(tour1)
  elif method == 3:
    tour2 = ApplyInsertion(tour1)
  
  return tour2


from model import NN

def CreateAndTrainANN(x, y):
  nn = NN([29, 10, 1])
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
  _, _, train_error = nn.train_lm(x_train, y_train)
  _, _, test_error = nn.train_lm(x_test, y_test)
  
  results = {'TrainData' : {'E': train_error},
             'TestData': {'E': test_error}}
  
  return results

#tour -> for a particular ant

def FeatureSelectionCost(tour,no_of_selected_features,data):
  # Read Data Elements
  x = data['x']
  t = data['t']

  #Selected features
  Selected_features = tour[0 : no_of_selected_features]

  #Ratio of Selected Features
  ratio_features = no_of_selected_features/len(tour)

  #Selecting Features
  xs = []
  for i in Selected_features:
    xs.append(x[i])
  xs = np.asarray(xs)

  #Weights of Train and Test Errors
  wTrain = 0.8
  wTest = 1 - wTrain

  #Number of Runs
  nRun = 3
  EE = np.zeros((1, nRun))
  for r in range(0, nRun):
    #Create and Train ANN
    results = CreateAndTrainANN(xs, t);

    #Calculate Overall Error
    EE[0][r] = wTrain * results['TrainData']['E'] + wTest * results['TestData']['E']
  
  E = np.mean(EE)

  #Calculate Final Cost
  z = E

  #Set Outputs
  out = {}
  out['Selected_features'] = Selected_features
  out['no_of_selected_features'] = no_of_selected_features
  out['ratio_features'] = ratio_features
  out['E'] = E
  out['z'] = z

  return z, out


def ACO():
  #Problem Definition
  data = LoadData()

  #Desired Number of Selected Features
  no_of_selected_features = 4

  nVar = data['nx']

  #ACO Parameters
  MaxIt = 1  #Maximum Number of Iterations
  nAnt = 2   #Number of Ants (Population Size)
  Q = 1
  tau0 = 1    #Initial Phromone
  alpha = 1
  beta = 1
  rho = 0.05  #Evaporation Rate

  #Initialization
  eta = np.ones((nVar, nVar))         #Heuristic Information Matrix
  tau = tau0 * np.ones((nVar, nVar))  #Phromone Matrix
  BestCost = np.zeros((MaxIt, 1))     #Array to Hold Best Cost Values

  #Ant Colony Matrix
  ant = [[ {"Tour": [], "Cost": None, "Out": []} for y in range(1)] for x in range(nAnt)]

  #Best Ant
  BestAnt = {}
  BestAnt["Cost"] = float('inf')

  #ACO Main Loop
  for it in range(0, MaxIt):
    for k in range(0, nAnt):
      print("Ant: " + str(k))
      ant[k][0]["Tour"].append(random.randint(0, nVar))

      for l in range(1, nVar):
        print("Pos of nVar:" + str(l))
        i = ant[k][0]["Tour"][-1]  #Getting the last element of the Tour list
        P = np.multiply(np.power(tau[i:], alpha), np.power(eta[i:], beta))
        # print("Shape P : " + str(P.shape))
        # print("Ant k tour value: " + str(ant[k][0]["Tour"]))
        tour_no = ant[k][0]["Tour"][0]
        # print(tour_no)
        index = tour_no % P.shape[0]
        P[index] = 0
        P = P/np.sum(P, axis=0)
        # print(P)
        j = RouletteWheelSelection(P)
        ant[k][0]["Tour"].append(j)
      
      #Cost Function
      cost, out = FeatureSelectionCost(ant[k][0]["Tour"], no_of_selected_features, data)
      ant[k][0]["Cost"] = cost
      ant[k][0]["Out"].append(out)

      if ant[k][0]["Cost"] < BestAnt["Cost"]:
        BestAnt["best_ant"] = ant[k];

    #Update Phromones
    for k in range(0, nAnt):
      tour = ant[k][0]["Tour"]
      tour = tour.append(tour[0])

      for l in range(0, nVar):
        i = tour[l]
        j = tour[l+1]

        tau[i][j] = tau[i][j] + Q/ant[k][0]["Cost"]
    
    #Evaporation
    tau = (1-rho)*tau

    #Store Best Cost
    BestCost[it][0] = BestAnt["Cost"]

    #Show Iteration Information
    # print("Iteration: " + str(it) + "     BestCost: " + str(BestCost[it][0]))
    return BestAnt


@app.route("/food", methods=["POST"])
def index():
	BestANt = ACO()
	food_finder = BestAnt["best_ant"]["Out"]["Selected_features"]
    food = []
    food_data = scipy.io.loadmat("food.mat")
    for f in food_finder:
    	food.append(food_data[f][0])

    recommeded_food = {}
    recommeded_food['food'] = food
    return jsonify(recommeded_food)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)