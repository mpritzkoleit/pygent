import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import os
import pickle
import inspect
from shutil import copyfile
import copy
import time

# pygent
from pygent.agents import Agent
from pygent.data import DataSet
from pygent.environments import StateSpaceModel
from pygent.algorithms.core import Algorithm
from pygent.algorithms.nmpc import NMPC
from pygent.nn_models import NNDynamics


class MBRL(Algorithm):

    def __init__(self, environment, t, dt,
                 test_t=0.,
                 plotInterval=1,
                 nData=int(1e6),
                 path='../results/mbrl/',
                 checkInterval=2,
                 evalPolicyInterval=100,
                 warm_up_episodes=10,
                 dyn_lr=1e-3,
                 batch_size=512,
                 training_epochs=60,
                 data_ratio = 1,
                 aggregation_interval=1,
                 fcost=None,
                 horizon=None,
                 use_mpc=False,
                 ilqr_print=False,
                 ilqr_save=False,
                 ilqr_save_interval=100,
                 print_dyn_error=False,
                 weight_decay=1e-3,
                 data_noise=1e-3,
                 prediction_error_bound=1e-3,
                 ou_theta=1,
                 ou_sigma=0.2,
                 maxIters=500,
                 sparse_dyn=True):
        self.sparse_dyn = sparse_dyn
        xDim = environment.xDim
        oDim = environment.oDim
        uDim = environment.uDim
        uMax = environment.uMax
        if self.sparse_dyn:
            dxDim = 0.5*xDim
        else:
            dxDim = xDim
        if horizon == None or not use_mpc:
            horizon = t
        self.dt = dt

        self.nn_dynamics = NNDynamics(xDim, uDim,
                                      oDim=oDim,
                                      dxDim = dxDim,
                                      xIsAngle=environment.xIsAngle) # neural network dynamics
        self.optim = torch.optim.Adagrad(self.nn_dynamics.parameters(),
                                      lr=dyn_lr,
                                      weight_decay=weight_decay)
        self.nn_environment = StateSpaceModel(self.ode, environment.cost, environment.x0, uDim, dt)
        self.nn_environment.uMax = uMax
        self.nmpc_algorithm = NMPC(environment, self.nn_environment, t, dt, horizon,
                                   init_optim=False,
                                   add_noise=True,
                                   path=path,
                                   fcost=fcost,
                                   fastForward=True,
                                   maxIters=maxIters,
                                   step_iterations=1,
                                   ilqr_print=ilqr_print,
                                   ilqr_save=ilqr_save,
                                   tolFun=1e-4,
                                   save_interval=ilqr_save_interval,
                                   noise_gain=0.005,
                                   ou_theta=ou_theta,
                                   ou_sigma=ou_sigma)
        super(MBRL, self).__init__(environment, self.nmpc_algorithm.agent, t, dt)
        self.D_rand = DataSet(nData)
        self.D_RL = DataSet(nData)
        self.test_t = test_t
        self.plotInterval = plotInterval  # inter
        self.evalPolicyInterval = evalPolicyInterval
        self.checkInterval = checkInterval  # checkpoint interval
        self.path = path
        self.warm_up = warm_up_episodes*int(t/dt)
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.data_ratio = data_ratio
        self.aggregation_interval = aggregation_interval
        self.use_mpc = use_mpc
        self.dynamics_first_trained = False
        self.print_dyn_error = print_dyn_error
        self.data_noise = data_noise
        self.prediction_error_bound = prediction_error_bound
        self.training_time = []
        self.optim_time = []


        if not os.path.isdir(path):
            os.makedirs(path)
        if not os.path.isdir(path + 'plots/'):
            os.makedirs(path + 'plots/')
        if not os.path.isdir(path + 'animations/'):
            os.makedirs(path + 'animations/')
        if not os.path.isdir(path + 'data/'):
            os.makedirs(path + 'data/')
        copyfile(inspect.stack()[-1][1], path + 'exec_script.py')
        self.expCost = []
        self.episode_steps = []

    def ode(self, t, x, u):
        if self.sparse_dyn:
            rhs = np.concatenate((x[int(len(x) / 2):], 1 / self.dt * self.nn_dynamics.ode(x, u)))
        else:
            rhs = 1/self.dt*self.nn_dynamics.ode(x, u)
        return rhs

    def run_episode(self, reinit=True):
        """ Run a training episode. If terminal state is reached, episode stops."""

        print('Started episode ', self.episode)
        tt = np.arange(0, self.t+self.test_t, self.dt)
        cost = []  # list of incremental costs
        disc_cost = [] # discounted cost
        start_time = time.time()
        if self.use_mpc:
            self.agent.init_optim(self.environment.x0)
        else:
            self.agent.init_optim(self.environment.x0, init=True)
        self.optim_time.append((time.time() - start_time))
        self.environment.reset(self.environment.x0)
        self.agent.reset()


        for i, t in enumerate(tt):
            # agent computes control/action
            if self.use_mpc:
                u = self.agent.take_action(self.dt, self.environment.x)
            else:
                u = self.agent.take_action_plan_feedback(self.dt, self.environment.x, i)

            # simulation of environment
            c = self.environment.step(u, self.dt)
            cost.append(c)
            disc_cost.append(c)

            # store transition in data set (x_, u, x, c)
            transition = ({'x_': self.environment.x_ + np.random.normal(0, self.data_noise, self.environment.xDim),
                           'u': self.agent.u + np.random.normal(0, self.data_noise, self.environment.uDim),
                           'x': self.environment.x + np.random.normal(0, self.data_noise, self.environment.xDim),
                           'o_': self.environment.o_ + np.random.normal(0, self.data_noise, self.environment.oDim),
                           'o': self.environment.o + np.random.normal(0, self.data_noise, self.environment.oDim),
                           'c': [c],
                           't': [self.environment.terminated]})

            prediction_loss = self.pred_loss(transition)
            print(prediction_loss)
            # add sample to data set
            if prediction_loss > self.prediction_error_bound:
                # only add sample, if prediction error is higher than error-bound
                self.D_RL.force_add_sample(transition)

            # check if environment terminated
            if self.environment.terminated:
                print('Environment terminated!')
                break
        self.test_dynamics(self.environment.history[0])
        # store the mean of the incremental cost
        self.meanCost.append(np.mean(cost))
        self.totalCost.append(np.sum(disc_cost))
        self.episode_steps.append(i)
        self.episode += 1
        pass

    def pred_loss(self, transition):
        x_Inputs = torch.Tensor([transition['x_']])
        # x_Inputs_norm = (x_Inputs - self.nn_dynamics.xMean) / self.nn_dynamics.xVar

        xInputs = torch.Tensor([transition['x']])
        # xInputs_norm = (xInputs - self.nn_dynamics.xMean) / self.nn_dynamics.xVar

        o_Inputs = torch.Tensor([transition['o_']])
        o_Inputs_norm = (o_Inputs - self.nn_dynamics.oMean) / self.nn_dynamics.oVar

        uInputs = torch.Tensor([transition['u']])
        uInputs_norm = (uInputs - self.nn_dynamics.uMean) / self.nn_dynamics.uVar

        dx = xInputs - x_Inputs
        if self.sparse_dyn:
            dx = dx.split(int(dx.shape[1]/2), 1)[1]
        dx_norm = (dx - self.nn_dynamics.dxMean) / self.nn_dynamics.dxVar

        fOutputs_norm = self.nn_dynamics(o_Inputs_norm, uInputs_norm)
        prediction_loss = np.linalg.norm((fOutputs_norm-dx_norm).detach().numpy(), 2)
        return prediction_loss

    def run_random_episode(self):
        print('Warmup. Started episode ', self.episode)
        tt = np.arange(0, self.t, self.dt)
        cost = []  # list of incremental costs
        disc_cost = []  # discounted cost

        # reset environment/agent to initial state, delete history
        self.environment.reset(self.environment.x0)
        self.agent.reset()

        for i, t in enumerate(tt):
            # agent computes control/action
            u = self.agent.take_random_action(self.dt, ou_noise=False)
            # simulation of environment
            c = self.environment.step(u, self.dt)
            cost.append(c)
            disc_cost.append(c)

            # store transition in data set (x_, u, x, c)
            transition = ({'x_': self.environment.x_ + np.random.normal(0, self.data_noise, self.environment.xDim),
                           'u': self.agent.u + np.random.normal(0, self.data_noise, self.environment.uDim),
                           'x': self.environment.x + np.random.normal(0, self.data_noise, self.environment.xDim),
                           'o_': self.environment.o_+ np.random.normal(0, self.data_noise, self.environment.oDim),
                           'o': self.environment.o + np.random.normal(0, self.data_noise, self.environment.oDim),
                           'c': [c],
                           't': [self.environment.terminated]})

            # add sample to data set
            self.D_rand.force_add_sample(transition)
            # check if environment terminated
            if self.environment.terminated:
                print('Environment terminated!')
                break

        # store the mean of the incremental cost
        self.meanCost.append(np.mean(cost))
        self.totalCost.append(np.sum(disc_cost))
        self.episode_steps.append(i)
        self.episode += 1
        pass

    def run_controller(self, x0):
        """ Run an episode, where the policy network is evaluated. """

        print('Started episode ', self.episode)
        tt = np.arange(0, self.t+self.test_t, self.dt)
        cost = []  # list of incremental costs

        # reset environment/agent to initial state, delete history
        self.environment.reset(x0)
        self.agent.reset()

        for i, t in enumerate(tt):
            # agent computes control/action
            u = self.agent.take_action(self.dt, self.environment.x)

            # simulation of environment
            c = self.environment.step(u, self.dt)
            cost.append(c)

            # check if environment terminated
            if self.environment.terminated:
                print('Environment terminated!')
                break
        pass

    def run_learning(self, n):
        """ Learning process.

            Args:
                n (int): number of episodes
        """

        for steps in range(1, int(n) + 1):
            if self.D_rand.data.__len__()<max(self.batch_size, self.warm_up):#self.batch_size:
                self.run_random_episode()
            else:
                if not self.dynamics_first_trained:
                    self.train_dynamics()
                    self.dynamics_first_trained = True
                    self.run_episode()
                else:
                    if steps % self.aggregation_interval == 0 & self.dynamics_first_trained:
                        self.train_dynamics()
                        self.run_episode()
                    else:
                        self.run_episode()
                if not self.use_mpc:
                    self.agent.traj_optimizer.save()

            # plot environment after episode finished
            print('Samples: ', self.D_rand.data.__len__(), self.D_RL.data.__len__())
            if self.episode % 10 == 0:
                self.learning_curve()
            if self.episode % self.checkInterval == 0:
                self.save()
                # if self.meanCost[-1] < 0.01: # goal reached
            self.plot()
            #self.animation()
        pass

    def save(self):
        """ Save neural network parameters and data set. """

        # save network parameters
        torch.save({'nn_dynamics': self.nn_dynamics.state_dict()}, self.path + 'data/checkpoint'+ str(self.episode - 1)+'.pth')
        torch.save({'nn_dynamics': self.nn_dynamics.state_dict()}, self.path + 'data/checkpoint.pth')

        self.nn_dynamics.save_moments(str(self.episode - 1), self.path)

        # save data set
        self.D_rand.save(self.path + 'data/dataSet_D_rand.p')
        self.D_RL.save(self.path + 'data/dataSet_D_RL.p')
        self.D_RL.save(self.path + 'data/dataSet_D_RL'+ str(self.episode - 1) + '.p')
        # save learning curve data
        learning_curve_dict = {'totalCost': self.totalCost, 'meanCost':self.meanCost,
                               'expCost': self.expCost, 'episode_steps': self.episode_steps}

        with open(self.path + 'data/learning_curve.p', 'wb') as open_file:
            pickle.dump(learning_curve_dict, open_file)
        print('Network parameters, data set and learning curve saved.')
        with open(self.path + 'data/training_time.p', 'wb') as open_file:
            pickle.dump(self.training_time, open_file)
        with open(self.path + 'data/optim_time.p', 'wb') as open_file:
            pickle.dump(self.optim_time, open_file)
        print('Time logging saved.')
        pass

    def load(self, episode=''):
        """ Load neural network parameters and data set. """

        # load network parameters
        if os.path.isfile(self.path + 'data/checkpoint'+episode+'.pth'):
            checkpoint = torch.load(self.path + 'data/checkpoint'+episode+'.pth')
            self.nn_dynamics.load_state_dict(checkpoint['nn_dynamics'])
            print('Loaded neural network parameters!')
        else:
            print('Could not load neural network parameters!')

        if os.path.isfile(self.path + 'data/moments_dict'+episode+'.p'):
            self.nn_dynamics.load_moments(self.path)
            print('Loaded neural network moments!')
        else:
            print('Could not load neural network moments!')
        # load data set
        if os.path.isfile(self.path + 'data/dataSet_D_rand.p'):
            self.D_rand.load(self.path + 'data/dataSet_D_rand.p')
            print('Loaded data set D_rand!')
        else:
            print('No data set found!')

        # load data set
        if os.path.isfile(self.path + 'data/dataSet_D_RL'+episode+'.p'):
            self.D_RL.load(self.path + 'data/dataSet_D_RL'+episode+'.p')
            print('Loaded data set D_RL!')
        else:
            print('No data set found!')

        # load learning curve
        if os.path.isfile(self.path + 'data/learning_curve.p'):
            with open(self.path + 'data/learning_curve.p', 'rb') as open_file:
                learning_curve_dict = pickle.load(open_file)
            self.meanCost = learning_curve_dict['meanCost']
            self.totalCost = learning_curve_dict['totalCost']
            self.expCost = learning_curve_dict['expCost']
            self.episode_steps = learning_curve_dict['episode_steps']
            self.episode = self.meanCost.__len__() + 1
            print('Loaded learning curve data!')
        else:
            print('No learning curve data found!')
        #self.run_controller(self.environment.x0)
        if os.path.isfile(self.path + 'data/training_time.p'):
            with open(self.path + 'data/training_time.p', 'rb') as open_file:
            	self.training_time = pickle.load(open_file)
        if os.path.isfile(self.path + 'data/optim_time.p'):       
            with open(self.path + 'data/optim_time.p', 'rb') as open_file:
                self.optim_time = pickle.load(open_file)
        pass

    def plot(self):
        """ Plots the environment's and agent's history. """

        self.environment.plot()
        self.environment.save_history('env'+ str(self.episode - 1) , self.path + 'data/')
        plt.savefig(self.path + 'plots/' + str(self.episode - 1) + '_environment.pdf')
        self.agent.plot()
        self.agent.save_history('agent'+ str(self.episode - 1) , self.path + 'data/')
        plt.savefig(self.path + 'plots/' + str(self.episode - 1) + '_agent.pdf')
        plt.close('all')
        pass

    def animation(self):
        """ Animation of the environment (if available). """

        ani = self.environment.animation()
        if ani != None:
            try:
                ani.save(self.path + 'animations/' + str(self.episode - 1) + '_animation.mp4', fps=1 / self.dt)
            except:
                ani.save(self.path + 'animations/' + str(self.episode - 1) + '_animation.gif', fps=1 / self.dt)
        plt.close('all')
        pass

    def learning_curve(self):
        """ Plot of the learning curve. """

        fig, ax = plt.subplots(2, 1, dpi=150, sharex=True, figsize=(5.56, 3.44))

        #x = np.arange(1, self.episode)
        x = np.linspace(1, self.D_rand.data.__len__()+self.D_RL.data.__len__(), self.episode-1)
        x = np.cumsum(self.episode_steps)

        ax[0].step(x, self.meanCost, 'b', lw=1, label=r'$\frac{1}{N}\sum_{k=0}^N c_k$')
        ax[0].legend(loc='center', bbox_to_anchor=(1.15, .5), ncol=1, shadow=True)
        ax[0].grid(True)
        ax[0].ticklabel_format(axis='both', style='sci', scilimits=(-3,4), useMathText=True)
        ax[1].step(x, self.totalCost, 'b', lw=1, label=r'$\sum_{k=0}^N\gamma^k c_k$')
        ax[1].grid(True)
        ax[1].legend(loc='center', bbox_to_anchor=(1.15, .5), ncol=1, shadow=True)
        ax[1].ticklabel_format(axis='both', style='sci',scilimits=(-3,5), useMathText=True)
        plt.rc('font', family='serif')
        plt.xlabel('Samples')
        plt.tight_layout()
        plt.savefig(self.path + 'learning_curve.pdf')
        # todo: save learning curve data
        plt.close('all')
        pass

    def train_dynamics(self):
        start_time = time.time()
        training_data_set = copy.deepcopy(self.D_rand)
        for _ in range(self.data_ratio):
            training_data_set.data += self.D_RL.data
        self.training(training_data_set)
        # update models in NMPC controller
        if self.print_dyn_error:
            self.dynamics_error()
        self.training_time.append((time.time() - start_time))

    def training(self, dataSet):
        # loss function (mean squared error)
        criterion = nn.MSELoss()
        self.nn_dynamics.eval()
        self.set_moments(dataSet)
        # create training data/targets
        for epoch in range(self.training_epochs):
            running_loss = 0.0
            for iter, batch in enumerate(dataSet.shuffled_batches(self.batch_size)):
                dx, fOutputs = self.training_data(batch)
                # definition of loss functions
                loss = criterion(fOutputs, dx)
                # train
                self.optim.zero_grad()  # delete gradients
                loss.backward()  # error back-propagation
                self.optim.step()  # gradient descent step
                running_loss += loss.item()
                # self.eval() # eval mode on (batch normalization)
            print('NN dynamics training loss:', running_loss / max(1, iter))
        pass


    def training_data(self, batch):
        x_Inputs = torch.Tensor([sample['x_'] for sample in batch])
        #x_Inputs_norm = (x_Inputs - self.nn_dynamics.xMean) / self.nn_dynamics.xVar

        xInputs = torch.Tensor([sample['x'] for sample in batch])
        #xInputs_norm = (xInputs - self.nn_dynamics.xMean) / self.nn_dynamics.xVar

        o_Inputs = torch.Tensor([sample['o_'] for sample in batch])
        o_Inputs_norm = (o_Inputs - self.nn_dynamics.oMean) / self.nn_dynamics.oVar

        uInputs = torch.Tensor([sample['u'] for sample in batch])
        uInputs_norm = (uInputs - self.nn_dynamics.uMean) / self.nn_dynamics.uVar

        dx = xInputs - x_Inputs
        if self.sparse_dyn:
            dx = dx.split(int(dx.shape[1]/2), 1)[1]
        dx_norm =  (dx - self.nn_dynamics.dxMean) / self.nn_dynamics.dxVar

        fOutputs_norm = self.nn_dynamics(o_Inputs_norm, uInputs_norm)
        return dx_norm, fOutputs_norm

    def set_moments(self, dataset):
        x_Inputs = torch.Tensor([sample['x_'] for sample in dataset.data])
        self.nn_dynamics.xMean = x_Inputs.mean(dim=0, keepdim=True)
        self.nn_dynamics.xVar = x_Inputs.std(dim=0, keepdim=True)
        xInputs = torch.Tensor([sample['x'] for sample in dataset.data])
        dx = xInputs - x_Inputs
        if self.sparse_dyn:
            dx = dx.split(int(dx.shape[1]/2), 1)[1]
        self.nn_dynamics.dxMean = dx.mean(dim=0, keepdim=True)
        self.nn_dynamics.dxVar = dx.std(dim=0, keepdim=True)
        o_Inputs = torch.Tensor([sample['o_'] for sample in dataset.data])
        self.nn_dynamics.oMean = o_Inputs.mean(dim=0, keepdim=True)
        self.nn_dynamics.oVar = o_Inputs.std(dim=0, keepdim=True)
        uInputs = torch.Tensor([sample['u'] for sample in dataset.data])
        self.nn_dynamics.uMean = uInputs.mean(dim=0, keepdim=True)
        self.nn_dynamics.uVar = uInputs.std(dim=0, keepdim=True)
        pass

    def test_dynamics(self, x0):
        # simulate dynamics
        env = copy.deepcopy(self.environment)
        nn_env = copy.deepcopy(self.nn_environment)
        nn_env2 = copy.deepcopy(self.nn_environment)
        env.reset(x0)
        nn_env.reset(x0)
        loss = []
        for u in self.agent.history[1:]:
            env.step(u, self.dt)
            nn_env.fast_step(u, self.dt)
            nn_env2.reset(env.x_)
            nn_env2.fast_step(u, self.dt)
            loss.append((env.x - nn_env2.x)**2)

        fig, ax = plt.subplots(2,1)
        ax[0].plot(nn_env.tt, env.history)
        ax[0].plot(nn_env.tt, nn_env.history)
        ax[1].plot(self.agent.tt, self.agent.history)
        mse = (nn_env.history - env.history) ** 2
        plt.savefig(self.path + 'plots/'+str(self.episode)+'_test.pdf')
        env.save_history('test_env'+ str(self.episode - 1) , self.path + 'data/')
        nn_env.save_history('test_nn_env'+ str(self.episode - 1) , self.path + 'data/')
        plt.close('all')
        print('Trajectory Prediction Error: ', np.mean(mse))
        print('One Step Prediction Error: ', np.mean(loss))

