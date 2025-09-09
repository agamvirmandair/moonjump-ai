# MoonJump-AI  🚀  
### A Custom 2D Gymnasium Environment with Proximal Policy Optimization (PPO)

---

## 📌 Project Overview  
This project explores reinforcement learning by creating a custom 2D game environment in **Gymnasium** and training agents using **Proximal Policy Optimization (PPO)**.  

The game, **MoonJump**, challenges an agent-controlled robot to dodge incoming obstacles of varying speeds, sizes, and distances.  

**Key contributions:**  
- 🕹️ Custom Gymnasium environment (`MoonJump-v0`) built with Pygame  
- 🤖 Baseline PPO training using Stable Baselines3  
- 🧠 From-scratch PPO implementation in PyTorch for deeper understanding  
- 📊 Performance comparison between SB3 and custom PPO agent  

<br>

---

## 🎮 Environment Design  

**Action Space:**  
- `Discrete(2)` → `0 = do nothing`, `1 = jump`  

**Observation Space (vector of floats):**  
- Robot’s vertical position  
- Robot’s vertical velocity  
- Distance to nearest obstacle  
- Height of nearest obstacle  
- Obstacle’s horizontal speed  

<br>

---

## 🏆 Reward Structure  

- **+0.1 + score × 0.01** → survival reward per frame  
- **+1** → for every obstacle avoided  
- **−5 + (PLAYER_X − obstacle_x) × 0.01** → penalty on collision, with small progress bonus  

🔹 **Details:**  
- **−5 penalty** → strong negative signal on crash  
- **Progress bonus** → higher reward as agent passes obstacles  
- **Do nothing & crash** → strong penalty  
- **Jump & survive longer** → smaller penalty  

👉 This balance encourages **long-term survival strategies** rather than random jumping.  

**Termination:**  
- Robot collides with an obstacle  

<br>

---

## ⚙️ Training Setup  

**Algorithms:**  
- PPO (baseline + custom)  

**Libraries:**  
- Gymnasium  
- Stable Baselines3  
- PyTorch  

**Hyperparameters:**  
- Learning rate: `3e-4`  
- γ (discount factor): `0.99`  
- n-steps: `2048`  
- Clip range: `0.2`  

**Custom PPO Highlights:**  
- Advantage estimation  
- Policy clipping  
- Separate actor-critic networks  
- Training loop built from scratch with PyTorch  

<br>

---

## 📊 Results  

**Stable Baselines3 PPO**  
- Best average score ≈ **500 (max_len)**  
- More stable & sample efficient  

<p align="center">
  <img width="1332" height="443" alt="sb3_ppo_graph" src="https://github.com/user-attachments/assets/e1e57bc4-71b8-4dee-a9f9-00a4cae8ffa0" />
</p>  

**Custom PPO**  
- Best average score ≈ **100**  
- Slower, but shows correct learning dynamics  

<p align="center">
  <img width="640" height="480" alt="Agent1_ppo_graph" src="https://github.com/user-attachments/assets/db48c1b9-79b8-462c-8603-95572a59e5ab" />
</p>  

---


## 🎥 Demo

![MoonJump_gif](https://github.com/user-attachments/assets/3253f356-f594-4bbe-adcb-6e417be34793)


