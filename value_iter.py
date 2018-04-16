import numpy as np 
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv

env=GridworldEnv()


def value_iteration(env,discount_factor=1.0,theta=0.0001):
	
	V=np.zeros(env.nS)

	def one_step_lookahead(state,V):

		A=np.zeros(env.nA)
		for a in range(env.nA):
			for prob,next_state,reward,done in env.P[s][a]:
				A[a]+=prob*(reward+discount_factor*V[next_state])
		return A

	while True:
		delta=0

		for s in range(env.nS):

			A=one_step_lookahead(s,V)
			best_action_value=np.max(A)
			delta=max(delta,np.abs(best_action_value-V[s]))

			V[s]=best_action_value

		if delta<theta:
			break

	policy=np.zeros([env.nS,env.nA])

	for s in range(env.nS):
		A=one_step_lookahead(s,V)
		best_action=np.argmax(A)
		policy[s,best_action]=1.0

	return policy,V
policy,V=value_iteration(env)
print('policy',policy)

print('value function',V)
print('gridworld',V.reshape(env.shape))
	# for s in range(env.nS):
	# 	check=[]

	# 	v=0

	# 	for a,action_prob in enumerate(policy[s]):

	# 		delta=0

	# 		for prob,next_state,reward,done in env.P[s][a]:
	# 			v+=prob*action_prob*(reward+discount_factor*V[next_state])
	# 			check.append((V[next_state],a))
	# 			print(v)

	# 	V[s]=v

	# 	# print('check',check)
	# 	print(V)
	# 	# chosen_action=np.argmax(policy[s])
	# 	# best_action=np.argmax(check)



value_iteration(env)