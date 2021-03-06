import numpy as np 
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv
env=GridworldEnv()

def policy_eval(policy,env,discount_factor=1.0,theta=0.00001):
	V=np.zeros(env.nS)
	while True:
		delta=0
		for s in range(env.nS):
			v=0
			for a,action_prob in enumerate(policy[s]):

				for prob,next_state,reward,done in env.P[s][a]:

					v+=action_prob*prob*(reward+discount_factor*V[next_state])

			delta=max(delta,np.abs(v-V[s]))
			V[s]=v

		if delta<theta:
			break
	return np.array(V)

def policy_imrpovement(env,policy_eval=policy_eval,discount_factor=1.0):

	def one_step_lookahead(state,V):

		A=np.zeros(env.nA)
		for a in range(env.nA):
			for prob,next_state,reward,done in env.P[state][a]:
				A[a]+=prob*(reward+discount_factor*V[next_state])

		return A

	policy=np.ones([env.nS,env.nA])/(env.nA)

	while True:

		V=policy_eval(policy,env)
		policy_stable=True

		for s in range(env.nS):
			chosen_a=np.argmax(policy[s])
			action_values=one_step_lookahead(s,V)
			best_a=np.argmax(action_values)

			if chosen_a!=best_a:
				policy_stable=False
			policy[s]=np.eye(env.nA)[best_a]

		if policy_stable:
			return policy,V



policy,V=policy_imrpovement(env)
print('policy',policy)

print('value function',V)
print('gridworld',V.reshape(env.shape))



