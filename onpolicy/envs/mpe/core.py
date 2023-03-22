import numpy as np
import seaborn as sns

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None
        # physical angle
        self.phi = None  # 0-2pi
        # physical angular velocity
        self.p_omg = None 
        self.last_a = 0

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # index among all entities (important to set for distance caching)
        self.i = 0
        # name
        self.name = ''
        # properties:
        self.size = 0.0050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # entity can pass through non-hard walls
        self.ghost = False
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.max_angular = None
        self.accel = None
        # state: including internal/mental state p_pos, p_vel
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        # commu channel
        self.channel = None

    @property
    def mass(self):
        return self.initial_mass

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agent are adversary
        self.adversary = False
        # agent are dummy
        self.dummy = False
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state: including communication state(communication utterance) c and internal/mental state p_pos, p_vel
        self.state = AgentState()
        # action: physical action u & communication action c
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # zoe 20200420
        self.goal = None
        self.done = False

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.walls = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping（阻尼）
        self.damping = 0 # 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # cache distances between all agents (not calculated by default)
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None
        # zoe 20200420
        self.world_length = 200
        self.world_step = 0
        self.num_agents = 0
        self.num_landmarks = 0

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def calculate_distances(self):
        """
        cached_dist_vect: 类似图论矩阵, 记录i_a和i_b相对位置关系的向量
        """
        if self.cached_dist_vect is None:
            # initialize distance data structure
            self.cached_dist_vect = np.zeros((len(self.entities),
                                              len(self.entities),
                                              self.dim_p))
            # calculate minimum distance for a collision between all entities （size相加?）
            self.min_dists = np.zeros((len(self.entities), len(self.entities)))  # N*N数组，N为智能体个数
            for ia, entity_a in enumerate(self.entities):
                for ib in range(ia + 1, len(self.entities)):
                    entity_b = self.entities[ib]
                    min_dist = entity_a.size + entity_b.size
                    self.min_dists[ia, ib] = min_dist
                    self.min_dists[ib, ia] = min_dist
            # 实对称距离矩阵

        for ia, entity_a in enumerate(self.entities):
            for ib in range(ia + 1, len(self.entities)):
                entity_b = self.entities[ib]
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                self.cached_dist_vect[ia, ib, :] = delta_pos
                self.cached_dist_vect[ib, ia, :] = -delta_pos

        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)

        self.cached_collisions = (self.cached_dist_mag <= self.min_dists)  # bool

    def assign_agent_colors(self):
        n_dummies = 0
        if hasattr(self.agents[0], 'dummy'):
            n_dummies = len([a for a in self.agents if a.dummy])
        n_adversaries = 0
        if hasattr(self.agents[0], 'adversary'):
            n_adversaries = len([a for a in self.agents if a.adversary])
        n_good_agents = len(self.agents) - n_adversaries - n_dummies
        # r g b
        dummy_colors = [(0.25, 0.75, 0.25)] * n_dummies
        # sns.color_palette("OrRd_d", n_adversaries)
        adv_colors = [(0.75, 0.25, 0.25)] * n_adversaries
        # sns.color_palette("GnBu_d", n_good_agents)
        good_colors = [(0.25, 0.25, 0.75)] * n_good_agents
        colors = dummy_colors + adv_colors + good_colors
        for color, agent in zip(colors, self.agents):
            agent.color = color

    # landmark color
    def assign_landmark_colors(self):
        for landmark in self.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])

    # update state of the world
    def step(self):
        # zoe 20200420
        self.world_step += 1
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self.policy_agents)
        
        # gather forces applied to entities
        u = [None] * len(self.entities)  # 空数组
        # apply agent physical controls
        u = self.apply_action_force(u)
        # integrate physical state
        self.integrate_state(u)

        # # calculate and store distances between all entities
        # if self.cache_dists:
        #     self.calculate_distances()

    # gather agent action forces
    def apply_action_force(self, u):
        # set applied forces
        '''
        for adversary agents, u = [ar, at]; 
        for escaping agents, u = [Vx, Vy];
        '''
        for i, agent in enumerate(self.agents):
            u[i] = agent.action.u
        return u

    def Get_antiClockAngle(self, v1, v2):  # v1逆时针转到v2所需角度。范围：0-2pi
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

    def integrate_state(self, u):  # u:[[1*2]...] 1*2n
        target_agent = self.scripted_agents[0]
        for i, agent in enumerate(self.agents):            
            if agent.adversary == True:  # u = [ar, at]
                # [at, ar]转为[v,w]
                e_r = (target_agent.state.p_pos - agent.state.p_pos)/np.linalg.norm(target_agent.state.p_pos - agent.state.p_pos)
                e_t = np.array([e_r[0]*np.cos(np.pi/2)-e_r[1]*np.sin(np.pi/2), e_r[0]*np.sin(np.pi/2)+e_r[1]*np.cos(np.pi/2)])  # 逆时针旋转90
                a_r, a_t= e_r*u[i][0], e_t*u[i][1]
                a_ = a_r + a_t
                pos_vec = np.array([np.cos(agent.state.phi), np.sin(agent.state.phi)])
                delta_theta = self.Get_antiClockAngle(pos_vec, a_)
                if 0<delta_theta<=np.pi:
                    pass
                else:
                    delta_theta -= np.pi*2 
                v_ = agent.max_speed * np.linalg.norm(a_)  # 正
                w_ = agent.max_angular * delta_theta/(np.pi)  # 可正可负
                # update phi
                agent.state.p_omg = w_
                agent.state.phi += agent.state.p_omg*self.dt
                if agent.state.phi>2*np.pi:
                    agent.state.phi -= 2*np.pi
                phi_ = agent.state.phi
                # update acc
                new_v = np.array([v_*np.cos(phi_), v_*np.sin(phi_)])
                agent.state.last_a = v_ - np.linalg.norm(agent.state.p_vel)
                # update p_vel
                agent.state.p_vel = new_v
                # update p_pos
                agent.state.p_pos += agent.state.p_vel * self.dt
            else:  # u = [Vx, Vy]
                agent.state.p_vel = np.array([u[i][0], u[i][1]])
                agent.state.p_pos += agent.state.p_vel * self.dt
            # print("agent {} speed is {}".format(agent.i, np.linalg.norm(agent.state.p_vel)))