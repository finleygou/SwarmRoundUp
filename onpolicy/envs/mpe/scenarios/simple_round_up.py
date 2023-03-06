import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario

class Scenario(BaseScenario):
    
    def __init__(self) -> None:
        super().__init__()
        self.d_cap = 1 # 期望围捕半径

    # 设置agent,landmark的数量，运动属性。
    def make_world(self,args):
        world = World()
        # set any world properties first
        world.dim_c = 4
        num_good_agents = args.num_good_agents # 1
        num_adversaries = args.num_adversaries # 5
        num_agents = num_adversaries + num_good_agents
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):  # i 从0到5
            agent.name = i  # 'agent %d' % i
            agent.collide = True
            agent.silent = True if i > 0 else False
            agent.adversary = True if i < num_adversaries else False  # agent 0 1 2 3 4:adversary.  5: good
            agent.size = 0.075 if agent.adversary else 0.045
            agent.accel = 3.0 if agent.adversary else 3.0  # max acc
            agent.max_speed = 1.0 if agent.adversary else 1.3
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # properties and initial states for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.45, 0.95, 0.45]) if not agent.adversary else np.array([0.95, 0.45, 0.45])
            # agent.color -= np.array([0.3, 0.3, 0.3]) if agent.leader else np.array([0, 0, 0])
            if i == 0:
                agent.state.p_pos = np.array([-2, 0])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            elif i == 1:
                agent.state.p_pos = np.array([-1, 0])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            elif i == 2:
                agent.state.p_pos = np.array([0, 0])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            elif i == 3:
                agent.state.p_pos = np.array([1, 0])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            elif i == 4:
                agent.state.p_pos = np.array([2, 0])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            elif i == 5:
                rand_pos = np.random.uniform(0, 1, 2)  # 1*2的随机数组，范围0-1
                r_, theta_ = 2*rand_pos[0], np.pi*2*rand_pos[1]  # 半径为2，角度360，随机采样。圆域。
                agent.state.p_pos = np.array([r_*np.cos(theta_), 5+r_*np.cos(theta_)])  # (0,5)为圆心
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)

    def benchmark_data(self, agent, world):
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    def GetClockAngle(v1, v2):  # v1逆时针转到v2所需角度。范围：0-2pi
        # 2个向量模的乘积
        TheNorm = np.linalg.norm(v1)*np.linalg.norm(v2)
        # 叉乘
        rho = np.arcsin(np.cross(v1, v2)/TheNorm)
        # 点乘
        theta = np.arccos(np.dot(v1,v2)/TheNorm)
        if rho < 0:
            return np.pi*2 - theta
        else:
            return theta

    def GetAcuteAngle(v1, v2):  # 计算锐角
        cos_ = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        return np.arccos(cos_)

    '''
    返回左右邻居下标(论文中邻居的定义方式)和夹角
        agent: 当前adversary agent
        adversary: 所有adversary agents数组
        target: good agent
    '''
    def find_neighbors(self, agent, adversary, target):
        angle_list = []
        for adv in adversary:
            if adv == agent:
                continue
            angle_ = self.GetClockAngle(agent.state.pos-target.state.pos, adv.state.pos-target.state.pos)
            angle_list.append(angle_)
        min_angle = min(angle_list)
        max_angle = max(angle_list)
        min_index = angle_list.index(min_angle)
        max_index = angle_list.index(max_angle)

        return [max_index, min_index], np.pi*2 - max_angle, min_angle

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    # agent 和 adversary 分别的reward
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        #boundary_reward = -10 if self.outside_boundary(agent) else 0
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    # 逃逸目标的奖励：策略固定。势函数。此处暂时忽略
    def agent_reward(self, agent, world):
        '''    # Agents are rewarded based on minimum agent distance to each landmark
                rew = 0
                shape = False
                adversaries = self.adversaries(world)
                if shape:
                    for adv in adversaries:
                        rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
                if agent.collide:
                    for a in adversaries:
                        if self.is_collision(a, agent):
                            rew -= 5
                def bound(x):
                    if x < 0.9:
                        return 0
                    if x < 1.0:
                        return (x - 0.9) * 10
                    return min(np.exp(2 * x - 2), 10)  # 1 + (x - 1) * (x - 1)

                for p in range(world.dim_p):
                    x = abs(agent.state.p_pos[p])
                    rew -= 2 * bound(x)

                for food in world.food:
                    if self.is_collision(agent, food):
                        rew += 2
                rew += 0.05 * min([np.sqrt(np.sum(np.square(food.state.p_pos - agent.state.p_pos))) for food in world.food])

                return rew
            '''
        return 0

    # individual adversary award
    def adversary_reward(self, agent, world):  # agent here is adversary
        # Agents are rewarded based on individual position advantage
        r_step = 0
        target = self.good_agents(world)[0]  # moving target
        adversaries = self.adversaries(world)
        N_adv = len(adversaries)
        d_i = np.linalg.norm(agent.state.p_pos - target.state.p_pos) - self.d_cap
        d_list = [np.linalg.norm(adv.state.p_pos - target.state.p_pos) - self.d_cap for adv in adversaries]   # left d for all adv
        d_mean = np.mean(d_list)
        sigma_d = np.std(d_list)
        exp_alpha = np.pi*2/N_adv
        # find neighbors (方位角之间不存在别的agent)
        _, left_nb_angle, right_nb_angle = self.find_neighbors(agent, adversaries, target)  # nb:neighbor
        delta_alpha = abs(left_nb_angle - right_nb_angle)
        # find min d between allies
        d_min = 20
        for adv in adversaries:
            if adv == agent:
                continue
            d_ = np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
            if d_ < d_min:
                d_min = d_

        k1, k2, k3, k4, k5 = 0.002, 0.5, 0.05, 0.1, 0.05
        w1, w2, w3, w4, w5 = 0.3, 0.1, 0.15, 0.25, 0.1

        r1 = -k1*d_i
        r2 = np.exp(-k2*(d_i - d_mean)/sigma_d) - 1
        r3 = min(np.exp(-k3*abs(left_nb_angle-exp_alpha)), np.exp(-k3*abs(right_nb_angle-exp_alpha))) - 1
        r4 = np.exp(-k4*delta_alpha) - 1
        r5 = np.exp(-k5*d_min)
        r_step = w1*r1+w2*r2+w3*r3+w4*r4+w5*r5

        return r_step

    # observation for adversary agents
    def observation(self, agent, world):
        target = self.good_agents(world)[0]  # moving target
        adversaries = self.adversaries(world)
        # calculate o_loc：
        Q_iM = self.GetAcuteAngle(target.state.p_pos-agent.state.p_pos, agent.state.p_vel)
        Q_iM_dot = self.GetAcuteAngle(target.state.p_vel-agent.state.p_vel, agent.state.last_a)  # 这样算应该是错误的。。
        dist_vec = agent.state.p_pos - target.state.p_pos
        vel_vec = agent.state.p_vel - target.state.p_vel
        d_i = np.linalg.norm(dist_vec) - self.d_cap
        d_i_dot = (dist_vec[0]*vel_vec[0]+dist_vec[1]*vel_vec[1])/np.linalg.norm(dist_vec)
        o_loc = [Q_iM, Q_iM_dot, d_i, d_i_dot, agent.state.p_vel, agent.state.last_a]
        # calculate o_ext：
        _, left_nb_angle, right_nb_angle = self.find_neighbors(agent, adversaries, target)  # nb:neighbor
        delta_alpha = abs(left_nb_angle - right_nb_angle)
        d_list = [np.linalg.norm(adv.state.p_pos - target.state.p_pos) - self.d_cap for adv in adversaries]   # left d for all adv
        d_mean = np.mean(d_list)
        o_ext = [delta_alpha, d_mean]
        # communication of all other agents
        o_ij = np.array([])
        for adv_j in adversaries:
            if adv_j is agent: continue
            d_ij = np.linalg.norm(agent.state.p_pos - adv_j.state.p_pos)
            Q_ij = self.GetAcuteAngle(adv_j.state.p_pos - agent.state.p_pos, agent.state.p_vel)
            Q_ji = self.GetAcuteAngle(adv_j.state.p_vel, agent.state.p_pos - adv_j.state.p_pos)
            dist_j_vec = adv_j.state.p_pos - target.state.p_pos
            d_j = np.linalg.norm(dist_j_vec) - self.d_cap
            delta_d_ij = d_i - d_j
            alpha_ij = self.GetAcuteAngle(dist_vec, dist_j_vec)
            o_ij = np.concatenate([o_ij]+[np.array([d_ij, Q_ij, Q_ji, delta_d_ij, alpha_ij])])
        
        return np.concatenate([o_loc] + [o_ext] + [o_ij])  # concatenate要用两层括号