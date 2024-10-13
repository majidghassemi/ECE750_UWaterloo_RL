from maze_env import Maze
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import time
import warnings
from RL_brainsample_wrong import rlalgorithm as rlalg1 
from RL_brainsample_qlearning import rlalgorithm as rlalg2
from RL_brainsample_doubqlearning import rlalgorithm as rlalg3
from RL_brainsample_sarsa import rlalgorithm as rlalg4
from RL_brainsample_expsarsa import rlalgorithm as rlalg5

DEBUG=1
def debug(debuglevel, msg, **kwargs):
    if debuglevel <= DEBUG:
        if 'printNow' in kwargs:
            if kwargs['printNow']:
                print(msg)
        else:
            print(msg)


def plot_rewards(experiments, window=100):
    """Generate plot of rewards for given experiments list and window controls the length
    of the window of previous episodes to use for average reward calculations.
    """
    plt.figure(2)
    plt.subplot(121)
    window_color_list=['blue','red','green','black','purple']
    color_list=['lightblue','lightcoral','lightgreen', 'darkgrey', 'magenta']
    label_list=[]
    for i, (name, env, RL, data) in enumerate(experiments):
        x_values=range(len(data['global_reward']))
        label_list.append(RL.display_name)
        y_values=data['global_reward']
        plt.plot(x_values, y_values, c=color_list[i],label=label_list[-1])
        plt.legend(label_list)
    if len(x_values) >= window : 
        for i, (name, env, RL, data) in enumerate(experiments):
            x_values=range(window, 
                    len(data['med_rew_window'])+window)
            y_values=data['med_rew_window']
            plt.plot(x_values, y_values,
                    c=window_color_list[i])
    plt.title("Summed Reward", fontsize=16)
    plt.xlabel("Episode", fontsize=16)
    plt.ylabel("Reward", fontsize=16)
    plt.tick_params(axis='both', which='major',
                    labelsize=14)
    #plt.show()

def plot_length(experiments):
    plt.figure(2)
    plt.subplot(122)
    color_list=['blue','green','red','black','magenta']
    label_list=[]
    for i, (name, env, RL, data) in enumerate(experiments):
        x_values=range(len(data['ep_length']))
        label_list.append(RL.display_name)
        y_values=data['ep_length']
        plt.plot(x_values, y_values, c=color_list[i],label=label_list[-1])
        plt.legend(label_list)
    plt.title("Path Length", fontsize=16)
    plt.xlabel("Episode", fontsize=16)
    plt.ylabel("Length", fontsize=16)
    plt.tick_params(axis='both', which='major',
                    labelsize=14)


def update(env, RL, data, episodes=50, window=100, **kwargs):
    global_reward = np.zeros(episodes)
    data['global_reward']=global_reward
    ep_length = np.zeros(episodes)
    data['ep_length']=ep_length
    if episodes >= window:
        med_rew_window = np.zeros(episodes-window)
        var_rew_window = np.zeros(episodes)
    else:
        med_rew_window = []
        var_rew_window = []
    data['med_rew_window'] = med_rew_window
    data['var_rew_window'] = var_rew_window

    for episode in range(episodes):  
        t=0
        ''' initial state
            Note: the state is represented as two pairs of 
            coordinates, for the bottom left corner and the 
            top right corner of the agent square.
        '''
        if episode == 0:
            state = env.reset(value = 0)
        else:
            state = env.reset()

        debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))

        if(showRender and (episode % renderEveryNth)==0):
            print('Rendering Now Alg:{} Ep:{}/{} at speed:{}'.format(RL.display_name, episode, episodes, sim_speed))

        # The main loop of the training on an episode
        # RL choose action based on state
        action = RL.choose_action(str(state))

        while True:
            # fresh env
            if(showRender and (episode % renderEveryNth)==0):
                env.render(sim_speed)

            # RL take action and get next state and reward
            state_, reward, done = env.step(action)
            global_reward[episode] += reward
            debug(2,'state(ep:{},t:{})={}'.format(episode, t, state))
            debug(2,'reward={:.3f} return_t={:.3f} Mean50={:.3f}'.format(reward, global_reward[episode],np.mean(global_reward[-50:])))

            # RL learn from this transition
            # and determine next state and action
            state, action =  RL.learn(str(state), action, reward, str(state_))

            # break while loop when end of this episode
            if done:
                break
            else:
                t=t+1

        debug(1,"({}) Ep {} Length={} Summed Reward={:.3} ".format(RL.display_name,episode, t,  global_reward[episode],global_reward[episode]),printNow=(episode%printEveryNth==0))

        #save data about length of the episode
        ep_length[episode]=t

        if(episode>=window):
            med_rew_window[episode-window] = np.median(global_reward[episode-window:episode])
            var_rew_window[episode-window] = np.var(global_reward[episode-window:episode])
            debug(1,"    Med-{}={:.3f} Var-{}={:.3f}".format(
                    window,
                    med_rew_window[episode-window],
                    window,
                    var_rew_window[episode-window]),
                printNow=(episode%printEveryNth==0))
    print('Algorithm {} completed'.format(RL.display_name))
    env.destroy()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="The frame\.append method is deprecated.*")
    #sim_speed of .1 is nice to view, .001 is fast but visible, sim_speed has not effect if showRender is False
    sim_speed = 0.008 #.001

    #Which task to run, select just one
    usetask = 3 # 1,2,3

    #Example Short Fast start parameters for Debugging
    showRender=True # True means renderEveryNth episode only, False means don't render at all
    episodes=2000 #100, 500, 1000
    renderEveryNth=10 #10, 100, 250 
    printEveryNth=1 #10, 25, 100
    window=3 #10, 25
    do_plot_rewards=True
    do_plot_length=True

    #Example Full Run, you may need to run longer
    #showRender=False
    #episodes=1000
    #renderEveryNth=100
    #printEveryNth=20
    #window=100
    #do_plot_rewards=True
    #do_plot_length=True

    if(len(sys.argv)>1):
        episodes = int(sys.argv[1])
    if(len(sys.argv)>2):
        showRender = sys.argv[2] in ['true','True','T','t']
    if(len(sys.argv)>3):
        datafile = sys.argv[3]


    # Task Specifications
    # point [0,0] is the top left corner
    # point [x,y] is x columns over and y rows down
    # range of x and y is [0,9]
    # agentXY=[0,0] # Agent start position [column, row]
    # goalXY=[4,4] # Target position, terminal state

    #Task 1
    if usetask == 1:
        agentXY=[1,6] # Agent start position
        goalXY=[8,1] # Target position, terminal state
        wall_shape=np.array([[2,6], [2,5], [2,4], [6,1],[6,2],[6,3]])
        pits=np.array([[9,1],[8,3], [0,9]])

    #Task 2 - cliff face
    if usetask == 2:
        agentXY=[0,2] # Agent start position
        goalXY=[2,6] # Target position, terminal state
        wall_shape=np.array([ [0,3], [0,4], [0,5], [0,6], [0,7], [0,1],[1,1],[2,1],[8,7],[8,5],[8,3],[2,7]])
        pits=np.array([[1,3], [1,4], [1,5], [1,6], [1,7], [2,5],[8,6],[8,4],[8,2]])


    #Task 3
    if usetask == 3:
        agentXY=[3,1] # Agent start position
        goalXY=[3,8] # Target position, terminal state
        wall_shape=np.array([[1,2],[1,3],[2,3],[4,3],[7,4],[3,6],[3,7],[2,7]])
        pits=np.array([[2,2],[3,4],[5,2],[0,5],[7,5],[0,6],[8,6],[0,7],[4,7],[2,8]])
    experiments=[]



    # First Demo Experiment 
    # Each combination of Algorithm and environment parameters constitutes an experiment that we will run for a number episodes, restarting the environment again each episode but keeping the value function learned so far.
    # You can add a new entry for each experiment in the experiments list and then they will all plot side-by-side at the end.
    experiments=[]
    
    # name1 = "WrongAlg on Task " + str(usetask)

    # env1 = Maze(agentXY,goalXY,wall_shape, pits, name1)
    # RL1 = rlalg1(actions=list(range(env1.n_actions)))
    # data1={}
    # env1.after(10, update(env1, RL1, data1, episodes, window))
    # env1.mainloop()
    # experiments.append((name1, env1,RL1, data1))

    name2 = "Q-Learning on Task " + str(usetask)
    env2 = Maze(agentXY, goalXY, wall_shape, pits, name2)
    RL2 = rlalg2(actions=list(range(env2.n_actions)))
    data2 = {}
    env2.after(10, update(env2, RL2, data2, episodes, window))
    env2.mainloop()
    experiments.append((name2, env2, RL2, data2))

    name3 = "Double Q-Learning on Task " + str(usetask)
    env3 = Maze(agentXY, goalXY, wall_shape, pits, name3)
    RL3 = rlalg3(actions=list(range(env3.n_actions)))
    data3 = {}
    env3.after(10, update(env3, RL3, data3, episodes, window))
    env3.mainloop()
    experiments.append((name3, env3, RL3, data3))

    name4 = "SARSA on Task " + str(usetask)
    env4 = Maze(agentXY, goalXY, wall_shape, pits, name4)
    RL4 = rlalg4(actions=list(range(env4.n_actions)))
    data4 = {}
    env4.after(10, update(env4, RL4, data4, episodes, window))
    env4.mainloop()
    experiments.append((name4, env4, RL4, data4))

    name5 = "Expected SARSA on Task " + str(usetask)
    env5 = Maze(agentXY, goalXY, wall_shape, pits, name5)
    RL5 = rlalg5(actions=list(range(env5.n_actions)))
    data5 = {}
    env5.after(10, update(env5, RL5, data5, episodes, window))
    env5.mainloop()
    experiments.append((name5, env5, RL5, data5))



    print("All experiments complete")

    # print(f"Experiment Setup:\n - episodes:{episodes} VI_sweeps:{VI_sweeps} sim_speed:{sim_speed}") 
    print(f"Experiment Setup:\n - episodes:{episodes}\n - sim_speed:{sim_speed}\n") 

    for name, env, RL, data in experiments:
        print("[{}] : {} : max-rew={:.3f} med-{}={:.3f} var-{}={:.3f} max-episode-len={}".format(
            name, 
            RL.display_name, 
            np.max(data['global_reward']),
            window,
            np.median(data['global_reward'][-window:]), 
            window,
            np.var(data['global_reward'][-window:]),
            np.max(data['ep_length'])))


    if(do_plot_rewards):
        #Simple plot of summed reward for each episode and algorithm, you can make more informative plots
        plot_rewards(experiments, window)

    if(do_plot_length):
        #Simple plot of summed reward for each episode and algorithm, you can make more informative plots
        plot_length(experiments)
        

    if(do_plot_rewards or do_plot_length):
        plt.figure(2)
        plt.suptitle("Task " + str(usetask), fontsize=20)
        plt.show()
        

