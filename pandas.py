import pandas as pd
import os
import matplotlib.pyplot as plt


dfs=[]
for eps in [0.01, 0.04, 0.07, 0.1, 0.15, 0.2, 0.25]:
    for fraction in  [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
        for noise in [0,1]:
            for tentative in [1,2,3,4]:
                if noise==1:
                    dir = str(eps)+'_'+str(fraction)+'_'+'noise'+'/'+str(tentative)
                else:
                    dir = str(eps)+'_'+str(fraction) + '/'+str(tentative)
                file_path=os.path.join('./log/', dir,'progress.csv')
                df=pd.read_csv(file_path,header=0)
                df=df.assign(eps=eps)
                df=df.assign(fraction=fraction)
                df=df.assign(noise=noise)
                df=df.assign(tentative=tentative)
                dfs.append(df)



data=pd.concat(dfs,sort=False,ignore_index=True)
data=data.drop('% time spent exploring',1)
data=data.drop('episodes',1)
'''res=data.groupby(['eps', 'fraction','step','noise','tentative'],as_index=False)[['mean 100 episode reward']].max()
res=res.groupby(['eps', 'fraction','noise'],as_index=False)[['mean 100 episode reward']].median()
res=res.groupby(['eps','noise'],as_index=False)[['mean 100 episode reward']].median()'''
steps=range(0,100001,1000)

plt.figure()
plt.title('Influence of parameter noise')
plt.ylabel('Reward')
plt.xlabel('Steps')
for eps in [0.01]:
    for fraction in  [0.1]:
        for noise in [0,1]:
            data_eps=data.where(data.get('eps')==eps)
            data_fraction=data_eps.where(data_eps.get('fraction')==fraction)
            data_noise=data_fraction.where(data_fraction.get('noise')==noise)

            data_noise=data_noise.dropna()
            step_reward_list=[0] * 101
            step_reward_list_max=[0] * 101
            step_reward_list_min=[999] * 101


            for tentative in [1,2,3,4]:
                data_sub=data_noise.where(data_noise.get('tentative')==tentative)
                data_sub = data_sub.dropna()

                for i,step in enumerate(steps):
                    # print(data_sub.get('steps'))
                    reward_step=data_sub.where(data_sub.get('steps')>step)
                    reward_step=reward_step.where(reward_step.get('steps')<step+1000)
                    reward_step=reward_step.dropna()
                    for tmp in reward_step.get('mean 100 episode reward'):
                        temp=tmp
                    step_reward_list[i]+=temp
                    step_reward_list_max[i]=max(temp,step_reward_list_max[i])
                    step_reward_list_min[i]=min(temp,step_reward_list_min[i])


            step_reward_list = [x / 4 for x in step_reward_list]

            if noise==1:
                plt.plot(steps,step_reward_list,label='with parameter noise')
                plt.fill_between(steps, step_reward_list_min, step_reward_list_max, alpha=0.1)

            else:
                plt.plot(steps,step_reward_list,label='with epsilon-greedy')
                plt.fill_between(steps, step_reward_list_min, step_reward_list_max, alpha=0.1)

plt.legend()

plt.savefig('cartpole_parameternoise.png')
plt.show()
plt.close()


plt.figure()
plt.ylabel('Reward')
plt.xlabel('Steps')
plt.title('Influence of final epsilon')
for eps in [0.01,0.1,0.25]:
    for fraction in  [0.1]:
        for noise in [0]:
            data_eps=data.where(data.get('eps')==eps)
            data_fraction=data_eps.where(data_eps.get('fraction')==fraction)
            data_noise=data_fraction.where(data_fraction.get('noise')==noise)

            data_noise=data_noise.dropna()
            step_reward_list=[0] * 101
            step_reward_list_max=[0] * 101
            step_reward_list_min=[999] * 101

            for tentative in [1,2,3,4]:
                data_sub=data_noise.where(data_noise.get('tentative')==tentative)
                data_sub = data_sub.dropna()


                for i,step in enumerate(steps):
                    # print(data_sub.get('steps'))
                    reward_step=data_sub.where(data_sub.get('steps')>step)
                    reward_step=reward_step.where(reward_step.get('steps')<step+1000)
                    reward_step=reward_step.dropna()
                    for tmp in reward_step.get('mean 100 episode reward'):
                        temp=tmp
                    step_reward_list[i]+=temp
                    step_reward_list_max[i] = max(temp, step_reward_list_max[i])
                    step_reward_list_min[i] = min(temp, step_reward_list_min[i])
            step_reward_list = [x / 4 for x in step_reward_list]


            plt.plot(steps,step_reward_list,label='final epsilon = '+str(eps))
            plt.fill_between(steps, step_reward_list_min, step_reward_list_max, alpha=0.1)

plt.legend()

plt.savefig('cartpole_eps.png')
plt.show()
plt.close()

plt.figure()
plt.ylabel('Reward')
plt.xlabel('Steps')
plt.title('Influence of exploration fraction')
for eps in [0.01]:
    for fraction in [0.1,0.2,0.3]:
        for noise in [0]:
            data_eps = data.where(data.get('eps') == eps)
            data_fraction = data_eps.where(data_eps.get('fraction') == fraction)
            data_noise = data_fraction.where(data_fraction.get('noise') == noise)

            data_noise = data_noise.dropna()
            step_reward_list = [0] * 101
            step_reward_list_max=[0] * 101
            step_reward_list_min=[999] * 101
            for tentative in [1, 2, 3, 4]:
                data_sub = data_noise.where(data_noise.get('tentative') == tentative)
                data_sub = data_sub.dropna()

                for i, step in enumerate(steps):
                    # print(data_sub.get('steps'))
                    reward_step = data_sub.where(data_sub.get('steps') > step)
                    reward_step = reward_step.where(reward_step.get('steps') < step + 1000)
                    reward_step = reward_step.dropna()
                    for tmp in reward_step.get('mean 100 episode reward'):
                        temp = tmp
                    step_reward_list[i] += temp
                    step_reward_list_max[i] = max(temp, step_reward_list_max[i])
                    step_reward_list_min[i] = min(temp, step_reward_list_min[i])
            step_reward_list = [x / 4 for x in step_reward_list]

            plt.plot(steps, step_reward_list, label='exploration fraction = ' + str(fraction))
            plt.fill_between(steps, step_reward_list_min, step_reward_list_max, alpha=0.1)

plt.legend()
plt.savefig('cartpole_fraction.png')
plt.show()
plt.close()
