import gymnasium as gym
import numpy as np
import pygame
import random
from gymnasium.envs.registration import register

register(
    id="MoonJumpEnv-v0",
    entry_point="MoonJump_env:MoonJumpEnv",
)

WHITE = (255,255,255)
PURPLE = (106,67,141)

WIDTH = 1000
HEIGHT = 750
PLAYER_FLOOR = 620  
STARS = [[random.randint(0, WIDTH), random.randint(0, HEIGHT)] for _ in range(40)]
PLAYER_X = 150
JUMP_STRENGTH = 18
GRAVITY = 0.55
PLAYER_SPEED = 7
FPS = 60
TALL_OBSTACLE_DIM = (150,175)
SHORT_OBSTACLE_DIM = (200,100)
MIN_SPEED = 7
MAX_SPEED = 12

def normalize(val, min_val, max_val):
    return np.clip(2 * (val - min_val) / (max_val - min_val) - 1, -1, 1)

class MoonJumpEnv(gym.Env):
    def __init__(self, render_mode = None):
        
        super().__init__()
        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("MoonJump")
            self.font = pygame.font.SysFont('bauhaus93', 80)
            self.clock = pygame.time.Clock()
        else:
            self.screen = pygame.Surface((WIDTH, HEIGHT))
        self.robot_img = pygame.transform.smoothscale(pygame.image.load('Robot.png'), (100, 250))
        self.obstacle_variants = [pygame.transform.smoothscale(pygame.image.load('Obstacle1.png'), TALL_OBSTACLE_DIM),
                             pygame.transform.smoothscale(pygame.image.load('Obstacle2.png'), SHORT_OBSTACLE_DIM)]
        
        self.action_space = gym.spaces.Discrete(2)


        self.observation_space = gym.spaces.Box(
            low = np.array([-1,  -1, 0, -1, -1], dtype=np.float32),
            high = np.array([ 1,  1,  1,  1, 1], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)
        self.y_change = 0
        self.player_y = PLAYER_FLOOR
        self.score = 0
        self.obstacles_pos = [1000,2750]
        self.obstacle_speeds = [round(np.random.uniform(MIN_SPEED,MAX_SPEED),2),
                                round(np.random.uniform(MIN_SPEED,MAX_SPEED),2)]
        self.obstacles_heights = [np.random.choice([TALL_OBSTACLE_DIM[1],SHORT_OBSTACLE_DIM[1]]),
                                  np.random.choice([TALL_OBSTACLE_DIM[1],SHORT_OBSTACLE_DIM[1]])]
        self.obstacles_imgs = [np.random.choice(self.obstacle_variants), random.choice(self.obstacle_variants)]
        self.craters = [100, 300, 700, 850]
        self.active = True

        if self.render_mode == "human":
            self.timer = pygame.time.Clock()
        return self._get_obs1(), {}

    def step(self, action):
        prev_score = self.score
        self.spawn_obstacles()
        if self.render_mode == "human":
            self.timer.tick(FPS)
            self.draw_background()
            self.draw_objects()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.active = False

        if action == 1 and self.y_change == 0:
            self.y_change = JUMP_STRENGTH

        if self.active:
            self.update_obstacles()
            self.update_player()
            if self.render_mode == "human":
                self.update_craters()
                pygame.display.flip()
        terminated = not self.active
        truncated = False

        reward = 0.1 + 0.01*self.score
        if action == 1 and prev_score == self.score: 
            reward = -0.1 
        if terminated:
            reward = -5 + (PLAYER_X - self.obstacles_pos[self.incoming_obstacle])*0.01
        if self.score > prev_score: 
            reward = 1
        return self._get_obs1(), reward, terminated, truncated, {}


    def set_incoming_obstacle(self):
        if self.player_y == PLAYER_FLOOR:
            if self.obstacles_pos[0] > PLAYER_X and self.obstacles_pos[1] > PLAYER_X:
                self.incoming_obstacle = self.obstacles_pos.index(min(self.obstacles_pos))
            else:
                self.incoming_obstacle = self.obstacles_pos.index(max(self.obstacles_pos))

    def _get_obs1(self):
        y_norm = normalize(self.player_y, 316.4, PLAYER_FLOOR)
        y_change_norm = normalize(self.y_change, -JUMP_STRENGTH, JUMP_STRENGTH)
        self.set_incoming_obstacle()
        obstacle_height = 0 if self.obstacles_heights[self.incoming_obstacle] == SHORT_OBSTACLE_DIM[1] else 1
        obstacle_distance = normalize(self.obstacles_pos[self.incoming_obstacle]-PLAYER_X, -272.5, WIDTH*3)
        obstacle_speed = normalize(self.obstacle_speeds[self.incoming_obstacle], MIN_SPEED, MAX_SPEED)
        return np.array([y_norm, y_change_norm, obstacle_height, obstacle_distance, obstacle_speed], dtype=np.float32)

    def _get_obs2(self):
        y_norm = normalize(self.player_y, 316.4, PLAYER_FLOOR)
        y_change_norm = normalize(self.y_change, -JUMP_STRENGTH, JUMP_STRENGTH)
        obstacle_height0 = 0 if self.obstacles_heights[0] == SHORT_OBSTACLE_DIM[1] else 1
        obstacle_distance0 = normalize(self.obstacles_pos[0]-PLAYER_X, 0, WIDTH*4)
        obstacle_speed0 = normalize(self.obstacle_speeds[0], MIN_SPEED, MAX_SPEED)

        obstacle_height1 = 0 if self.obstacles_heights[1] == SHORT_OBSTACLE_DIM[1] else 1
        obstacle_distance1 = normalize(self.obstacles_pos[1]-PLAYER_X, 0, WIDTH*4)
        obstacle_speed1 = normalize(self.obstacle_speeds[1], MIN_SPEED, MAX_SPEED)
        return np.array([y_norm, y_change_norm, obstacle_height0, obstacle_distance0, obstacle_speed0,
                         obstacle_height1, obstacle_distance1, obstacle_speed1], dtype=np.float32)
    def update_craters(self):
        for x in range(4):
            self.craters[x] -= PLAYER_SPEED
            if self.craters[x] < -300: self.craters[x] = np.random.randint(1000, 1500)

    def set_obstacle_img(self, obstacle_ind):
        if self.obstacles_heights[obstacle_ind] == TALL_OBSTACLE_DIM[1]:
            self.obstacles_imgs[obstacle_ind] = self.obstacle_variants[0]
        elif self.obstacles_heights[obstacle_ind] == SHORT_OBSTACLE_DIM[1]:
            self.obstacles_imgs[obstacle_ind] = self.obstacle_variants[1]

    
    def obstacle_randomizer(self, obstacle_ind):
        self.obstacle_speeds[obstacle_ind] = round(np.random.uniform(MIN_SPEED,MAX_SPEED),2)
        self.obstacles_heights[obstacle_ind] = np.random.choice([TALL_OBSTACLE_DIM[1], SHORT_OBSTACLE_DIM[1]])
        speed_delta = self.obstacle_speeds[obstacle_ind] - self.obstacle_speeds[obstacle_ind-1]
        if speed_delta > 1:
            return 800 +speed_delta*round(np.random.uniform(250,300),2)
        return np.random.uniform(800, 900)
    
    def update_obstacles(self): 
        for i in  range(2):
            self.obstacles_pos[i] -= self.obstacle_speeds[i]
            if self.obstacles_pos[i] < -125:
                self.obstacles_pos[i] = self.obstacles_pos[1-i] + self.obstacle_randomizer(i)
                self.set_obstacle_img(i)
                #print(self.obstacles_pos[1-i], self.obstacle_speeds[1-i],  self.obstacles_pos[i], self.obstacle_speeds[i])
                self.score += 1
            if self.robot_rect.colliderect(self.obstacle0) or self.robot_rect.colliderect(self.obstacle1):
                self.active = False
                #running = False

    def draw_background(self):
        self.screen.fill((10, 10, 30))
        for star in STARS: pygame.draw.circle(self.screen, WHITE, star, 2)
    
    def spawn_obstacles(self):
        self.obstacle0 = self.obstacles_imgs[0].get_rect()
        self.obstacle0.center = (self.obstacles_pos[0], 745-0.5*self.obstacle0.h)

        self.obstacle1 = self.obstacles_imgs[1].get_rect()
        self.obstacle1.center = (self.obstacles_pos[1], 745-0.5*self.obstacle1.h )

        self.robot_rect = self.robot_img.get_rect()
        self.robot_rect.center = (PLAYER_X, self.player_y)
        


    
    def draw_objects(self):
        pygame.draw.polygon(self.screen, PURPLE, [(self.craters[0],700), (self.craters[0]+90, 700), (self.craters[0] +45, 680)])
        pygame.draw.polygon(self.screen, PURPLE, [(self.craters[1],700), (self.craters[1]+300, 700),(self.craters[1] +255, 680), (self.craters[1] +45, 680) ])
        pygame.draw.polygon(self.screen, PURPLE, [(self.craters[2],700), (self.craters[2]+ 90, 700), (self.craters[2]+ 45, 680)])
        pygame.draw.polygon(self.screen, PURPLE, [(self.craters[3],700), (self.craters[3]+ 110, 700), (self.craters[3]+ 55, 680)])

        pygame.draw.rect(self.screen, PURPLE, [0, 700 , WIDTH, HEIGHT-500])
        self.screen.blit(self.obstacles_imgs[0].convert_alpha(), self.obstacle0)
        self.screen.blit(self.obstacles_imgs[1].convert_alpha(), self.obstacle1)
        self.screen.blit(self.robot_img.convert_alpha(), self.robot_rect)
    
        score_text = self.font.render(f'{self.score}', True, WHITE)
        self.screen.blit(score_text, (450,150))

    def update_player(self):
            if self.y_change > 0 or self.player_y < PLAYER_FLOOR:
                self.player_y -= self.y_change
                self.y_change -= GRAVITY
            
            if self.player_y > PLAYER_FLOOR :
                self.player_y = PLAYER_FLOOR
            
            if self.player_y == PLAYER_FLOOR and self.y_change < 0:
                self.y_change = 0

if __name__ == "__main__":
    env = gym.make("MoonJumpEnv-v0", render_mode="human")
    obs, info = env.reset()
    for x in range(500):
        obs, reward, terminate, truncate, info = env.step(1)
        print(reward)