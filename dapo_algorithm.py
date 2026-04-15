import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import time
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import scipy.signal
from gymnasium.spaces import Box, Discrete
import os

# Import MPI tools from spinup which now handles fallback
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, MPI
from tqdm import tqdm

# # Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")\

device = torch.device("cpu")

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    """
    Magic from rllab for computing discounted cumulative sums of vectors.
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

# Custom sync_params function that handles CUDA tensors properly
def fixed_sync_params(module):
    """
    Sync parameters across MPI processes with proper CUDA handling
    """
    if num_procs() > 1:
        # Get local parameters as CPU tensors
        params = [p.detach().cpu() for p in module.parameters()]
        flat_params = torch.cat([p.reshape(-1) for p in params])
        
        # Gather parameters from all processes (on CPU)
        flat_params_np = flat_params.numpy()
        local_sum = np.zeros_like(flat_params_np)
        local_n = np.zeros_like(flat_params_np)
        local_sum[...] = flat_params_np
        local_n[...] = 1
        
        # MPI operations operate on numpy arrays
        comm = MPI.COMM_WORLD
        global_sum = np.zeros_like(local_sum)
        global_n = np.zeros_like(local_n)
        comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
        comm.Allreduce(local_n, global_n, op=MPI.SUM)
        
        # Compute the average
        avg_params_np = global_sum / global_n
        avg_params = torch.from_numpy(avg_params_np).float()
        
        # Update module parameters with the average values
        # And transfer back to the original device
        idx = 0
        for p in module.parameters():
            shape = p.shape
            numel = p.numel()
            new_p = avg_params[idx:idx+numel].reshape(shape).to(device)
            p.data.copy_(new_p)
            idx += numel

class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # Policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # Move to GPU
        self.to(device)

    def step(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
        return a.cpu().numpy(), logp_a.cpu().numpy()
    
    def act_batch(self, obs, num_samples=10):
        """Sample multiple actions for a single observation"""
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
            actions = []
            logps = []
            for _ in range(num_samples):
                pi = self.pi._distribution(obs)
                a = pi.sample()
                logp_a = self.pi._log_prob_from_distribution(pi, a)
                actions.append(a.cpu().numpy())
                logps.append(logp_a.cpu().numpy())
            return actions, logps

    def act(self, obs):
        return self.step(obs)[0]

class DAPOBuffer:
    """
    A buffer for storing trajectories experienced by a DAPO agent interacting
    with the environment. Includes dynamic sampling capability.
    """
    def __init__(self, obs_dim, act_dim, size, num_samples_per_state=10, gamma=0.99):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        
        # For group advantage calculation
        self.state_indices = np.zeros(size, dtype=np.int32)  # To track which state each sample belongs to
        self.num_samples_per_state = num_samples_per_state
        
        self.gamma = gamma
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.current_state_idx = 0

    def store(self, obs, act, rew, logp, state_idx):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.state_indices[self.ptr] = state_idx
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards to compute the return.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        
        # Compute returns
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def compute_group_advantages(self):
        """
        Compute group-relative advantages with dynamic sampling.
        Filter out state groups where all rewards are the same (all correct or all incorrect).
        """
        unique_states = np.unique(self.state_indices[:self.ptr])
        
        # Create a mask for states to keep
        states_to_keep = []
        
        for state_idx in unique_states:
            # Get indices for all samples from this state
            sample_indices = np.where(self.state_indices[:self.ptr] == state_idx)[0]
            
            if len(sample_indices) > 1:  # Need at least 2 samples to compute relative advantage
                # Get rewards for all samples from this state
                rewards = self.rew_buf[sample_indices]
                
                # Check if all rewards are the same (Dynamic Sampling)
                if np.std(rewards) > 1e-6:  # Not all rewards are the same
                    states_to_keep.append(state_idx)
                    
                    # Compute mean and std for normalization
                    mean_reward = np.mean(rewards)
                    std_reward = np.std(rewards)
                    
                    # Normalize rewards to get advantages
                    normalized_rewards = (rewards - mean_reward) / (std_reward + 1e-8)
                    
                    # Store normalized rewards as advantages
                    self.adv_buf[sample_indices] = normalized_rewards
        
        # Create a mask for data to keep
        mask = np.zeros(self.ptr, dtype=bool)
        for state_idx in states_to_keep:
            mask = mask | (self.state_indices[:self.ptr] == state_idx)
        
        return mask

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer. Also, resets some pointers in the buffer.
        Uses dynamic sampling to filter out states with all same rewards.
        """
        assert self.ptr > 0    # buffer has to have at least something in it
        
        # Compute group advantages and get mask for dynamic sampling
        mask = self.compute_group_advantages()
        
        # Get filtered data from buffer
        if np.any(mask):  # Check if we have any valid states
            data = dict(
                obs=self.obs_buf[:self.ptr][mask],
                act=self.act_buf[:self.ptr][mask],
                ret=self.ret_buf[:self.ptr][mask],
                adv=self.adv_buf[:self.ptr][mask],
                logp=self.logp_buf[:self.ptr][mask]
            )
        else:
            # If no valid states, return empty tensors
            data = dict(
                obs=np.zeros((0,) + self.obs_buf.shape[1:], dtype=np.float32),
                act=np.zeros((0,) + self.act_buf.shape[1:], dtype=np.float32),
                ret=np.zeros(0, dtype=np.float32),
                adv=np.zeros(0, dtype=np.float32),
                logp=np.zeros(0, dtype=np.float32)
            )
        
        # Reset pointers
        self.ptr, self.path_start_idx = 0, 0
        self.current_state_idx = 0
        
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

def dapo(env_fn,
         actor_critic=MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[256, 128], activation=torch.nn.ReLU),
         seed=42,
         steps_per_epoch=20000,
         epochs=100,
         gamma=0.995,
         epsilon_low=0.2,        # Lower clip ratio (DAPO specific)
         epsilon_high=0.28,      # Higher clip ratio (DAPO specific)
         pi_lr=3e-5,
         train_pi_iters=100,
         lam=0.95,
         max_ep_len=3000,
         target_kl=0.15,
         logger_kwargs=dict(),
         save_freq=10,
         num_samples_per_state=10,  # Number of action samples per state for group advantage
         env_kwargs=None,
         adjustment_type='both',   # Type of LLM adjustment: 'both', 'sentiment', 'risk', 'none'
         alpha=1.0,                # Exponent for sentiment adjustment
         beta=1.0,                 # Exponent for risk adjustment
         force_cpu=False):         # Force CPU usage
    
    # If force_cpu is True, ensure we're using CPU
    global device
    if force_cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage for DAPO algorithm")
    
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor (policy)
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Use our fixed sync params function that properly handles CUDA tensors
    fixed_sync_params(ac)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.pi])
    logger.log('\nNumber of parameters: \t pi: %d\n'%var_counts)

    # Set up experience buffer
    # Increase buffer size to accommodate multiple samples per state
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = DAPOBuffer(obs_dim, act_dim, local_steps_per_epoch * num_samples_per_state, num_samples_per_state, gamma)

    # Helper functions for hypothetical returns
    def calculate_portfolio_return(action, current_prices, next_prices):
        """Calculate the portfolio return for a given action (portfolio allocation)"""
        price_changes = next_prices / current_prices - 1
        return np.sum(action * price_changes)
    
    def extract_prices(state):
        """Extract prices from state"""
        stock_dim = env_kwargs["stock_dim"] if env_kwargs and "stock_dim" in env_kwargs else 84
        return state[0, 1:stock_dim+1]

    def extract_llm_features(state):
        """Extract LLM sentiment and risk scores from state"""
        stock_dim = env_kwargs["stock_dim"] if env_kwargs and "stock_dim" in env_kwargs else 84
        # State space structure: [Current Balance] + [Stock Prices] + [Stock Shares] + [Technical Indicators] + [LLM Sentiment] + [LLM Risk]
        # LLM Sentiment is before LLM Risk in the state space
        sentiment_start = -(2 * stock_dim)  # Second to last block
        risk_start = -stock_dim  # Last block
        
        llm_sentiments = state[0, sentiment_start:risk_start]
        llm_risks = state[0, risk_start:]
        
        return llm_sentiments, llm_risks

    # Set up function for computing DAPO policy loss with decoupled clipping
    def compute_loss_pi(data):
        obs = data['obs'].to(device)
        act = data['act'].to(device)
        adv = data['adv'].to(device)
        logp_old = data['logp'].to(device)

        # Policy loss with decoupled clipping (DAPO specific)
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        
        # Apply different clipping ranges for positive and negative advantages
        clip_low = 1.0 - epsilon_low
        clip_high = 1.0 + epsilon_high
        
        clip_ratio = torch.clamp(ratio, clip_low, clip_high)
        surrogate = ratio * adv
        clipped_surrogate = clip_ratio * adv
        
        # Minimize negative of minimum, so take min of surrogate and clipped surrogate
        loss_pi = -(torch.min(surrogate, clipped_surrogate)).mean()
        
        # DAPO removes the KL penalty - this is one of its key features
        total_loss = loss_pi

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(clip_high) | ratio.lt(clip_low)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return total_loss, pi_info

    # Set up optimizer for policy
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()
        
        # If we don't have any valid data (all states filtered by dynamic sampling)
        if data['obs'].shape[0] == 0:
            logger.log('No valid data for update (all states filtered by dynamic sampling)')
            return
        
        # Move data to the appropriate device
        data = {k: v.to(device) for k, v in data.items()}

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            
            # We need to move KL to CPU for MPI
            kl = pi_info['kl']
            # Use MPI to average KL across processes
            kl_avg = mpi_avg(kl)
            
            if kl_avg > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
                
            loss_pi.backward()
            
            # Custom gradient averaging with proper CUDA handling
            for p in ac.pi.parameters():
                if p.grad is not None:
                    # Move grad to CPU for MPI operations
                    p_grad_cpu = p.grad.detach().cpu().numpy()
                    # Average gradients across MPI processes
                    p_grad_avg = np.zeros_like(p_grad_cpu)
                    comm = MPI.COMM_WORLD
                    comm.Allreduce(p_grad_cpu, p_grad_avg, op=MPI.SUM)
                    # Copy back to GPU
                    p.grad.copy_(torch.from_numpy(p_grad_avg).to(device))
            
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    obs_reset = env.reset()
    o = obs_reset[0] if isinstance(obs_reset, tuple) else obs_reset
    ep_ret, ep_len = 0, 0
    state_idx = 0  # Track the state index for group advantage computation

    # Main loop: collect experience in env and update/log each epoch
    total_time_estimate = epochs * local_steps_per_epoch * 0.1  # rough estimate in seconds
    progress_bar = tqdm(total=epochs, desc="Training Progress", position=0)
    start_epoch_time = time.time()
    
    # Create directory for checkpoints
    checkpoint_dir = os.environ.get("DAPO_CHECKPOINT_DIR", "./checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        # Adjust the number of environment steps to account for multiple samples per state
        actual_env_steps = int(local_steps_per_epoch / num_samples_per_state)
        
        for t in range(actual_env_steps):
            # Get the current observation (state)
            current_state = o
            
            # Extract current prices for portfolio return calculation
            current_prices = extract_prices(current_state)
            
            # Sample multiple actions for the current state
            # In DAPO, we sample multiple actions for each state
            actions, logps = ac.act_batch(o, num_samples=num_samples_per_state)
            
            # Use the first action to step the environment
            step_result = env.step(actions[0])
            next_o, r = step_result[0], step_result[1]
            # Ensure r is a scalar float
            r = float(r.item() if hasattr(r, 'item') else r)
            
            # Support both 4-value (old gym) and 5-value (new gymnasium) step returns
            d = (step_result[2] or step_result[3]) if len(step_result) >= 5 else step_result[2]
            ep_ret += r
            ep_len += 1
            
            # Extract next state information
            next_state = next_o
            next_prices = extract_prices(next_state)
            
            # Extract sentiment and risk scores using the new helper function
            llm_sentiments, llm_risks = extract_llm_features(next_state)
            
            # Process all sampled actions
            for i in range(num_samples_per_state):
                action = actions[i]
                logp = logps[i]
                
                # For the first action, use the actual reward from the environment
                if i == 0:
                    base_reward = r
                else:
                    # For other actions, calculate hypothetical portfolio return
                    base_reward = calculate_portfolio_return(action, current_prices, next_prices)
                
                # Ensure base_reward is a scalar float
                base_reward = float(base_reward.item() if hasattr(base_reward, 'item') else base_reward)
                
                # Calculate position values based on this action and next prices
                position_values = action * next_prices
                total_value = np.sum(position_values)
                
                # Apply LLM risk and sentiment adjustment to reward following S_f/R_f formula
                if total_value == 0:
                    # If no positions, no adjustment
                    adjusted_reward = base_reward
                else:
                    # Continuous linear mapping for LLM scores (fixes TypeError for continuous values)
                    # Sentiment: [-1, 1] -> [0.99, 1.01] (1.0 is neutral)
                    llm_sentiment_weights = 1.0 + (llm_sentiments * 0.01)
                    # Risk: [0, 1] -> [0.99, 1.01] (0.5 is neutral)
                    llm_risks_weights = 1.0 + ((llm_risks - 0.5) * 0.02)
                    
                    # Calculate weights based on portfolio allocation
                    stock_weights = position_values / total_value
                    
                    # Calculate aggregated sentiment and risk
                    aggregated_sentiment = np.dot(stock_weights, llm_sentiment_weights)
                    aggregated_risk = np.dot(stock_weights, llm_risks_weights)
                    
                    # Apply reward adjustment based on the specified adjustment type and exponents
                    if adjustment_type == 'both':
                        # Use the r_{t,i}' = r_{t,i} u00d7 S_{f,i}^alpha/R_{f,i}^beta formula
                        adjustment_factor = (aggregated_sentiment ** alpha) / ((aggregated_risk ** beta) + 1e-8)
                        # Ensure adjustment_factor is a scalar
                        adjustment_factor = float(adjustment_factor.item() if hasattr(adjustment_factor, 'item') else adjustment_factor)
                        adjusted_reward = base_reward * adjustment_factor
                    elif adjustment_type == 'sentiment':
                        # Only use sentiment with exponent
                        adjusted_reward = base_reward * (aggregated_sentiment ** alpha)
                    elif adjustment_type == 'risk':
                        # Only use risk with exponent (inverse relationship)
                        adjusted_reward = base_reward / ((aggregated_risk ** beta) + 1e-8)
                    else:  # 'none'
                        # No adjustment
                        adjusted_reward = base_reward
                
                # Ensure logp is a scalar float
                logp = float(logp.item() if hasattr(logp, 'item') else logp)
                # Ensure action is squeezed to match act_dim (remove batch dim)
                action_to_store = np.squeeze(action)
                
                # Store in buffer with the same state index for all samples
                buf.store(current_state, action_to_store, adjusted_reward, logp, state_idx)
            
            # Move to next state index after collecting all samples for this state
            state_idx += 1
            
            # Update observation
            o = next_o
            
            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == actual_env_steps - 1
            
            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                
                # If trajectory didn't reach terminal state, bootstrap value
                if timeout or epoch_ended:
                    # For DAPO, we don't need a value estimate for bootstrapping
                    last_val = 0
                else:
                    last_val = 0
                    
                buf.finish_path(last_val)
                
                if terminal:
                    # Only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    
                # Reset for next episode
                obs_reset = env.reset()
                o = obs_reset[0] if isinstance(obs_reset, tuple) else obs_reset
                ep_ret, ep_len = 0, 0
        
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            checkpoint_path = os.path.join(checkpoint_dir, f"agent_dapo_deepseek_gpu_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': ac.state_dict(),
                'pi_optimizer_state_dict': pi_optimizer.state_dict(),
            }, checkpoint_path)
            logger.save_state({'env': env}, None)
        
        # Perform DAPO update
        update()
        
        # Calculate and update ETA
        epoch_time = time.time() - start_epoch_time
        eta = epoch_time * (epochs - epoch - 1)
        progress_bar.set_postfix({'ETA': f'{eta/3600:.1f}h'})
        progress_bar.update(1)
        start_epoch_time = time.time()
        
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
        
    progress_bar.close()
    
    # Save the final model
    final_model_path = os.path.join(checkpoint_dir, "agent_dapo_deepseek_gpu_final.pth")
    torch.save({
        'epoch': epochs-1,
        'model_state_dict': ac.state_dict(),
    }, final_model_path)
    print(f"\nTraining finished and final model saved in {final_model_path}")
    
    return ac

# Main execution
def run_dapo(env_train, env_kwargs, adjustment_type='both', alpha=1.0, beta=1.0):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=env_train)
    parser.add_argument('--hid', type=int, default=512)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--exp_name', type=str, default='dapo')
    parser.add_argument('-f', '--file', type=str, help='Kernel connection file')
    parser.add_argument('extra_args', nargs=argparse.REMAINDER)
    parser.add_argument('--cpu_only', action='store_true', help='Force CPU usage even if GPU is available')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID when multiple GPUs are available')
    args = parser.parse_args()
    
    # Use GPU by default unless --cpu_only is specified
    if torch.cuda.is_available() and not args.cpu_only:
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    elif args.cpu_only:
        print("Forcing CPU usage as requested")
        device = torch.device("cpu")
    
    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    # Number of action samples per state for DAPO
    num_samples_per_state = 10
    
    trained_dapo = dapo(
        lambda : args.env, 
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        seed=args.seed, 
        logger_kwargs=logger_kwargs,
        num_samples_per_state=num_samples_per_state,
        epochs=100,
        env_kwargs=env_kwargs,
        epsilon_low=0.2,      # DAPO specific parameter
        epsilon_high=0.28,    # DAPO specific parameter
        adjustment_type=adjustment_type,   # Type of LLM adjustment: 'both', 'sentiment', 'risk', 'none'
        alpha=alpha,                # Exponent for sentiment adjustment
        beta=beta                 # Exponent for risk adjustment
    )
    
    return trained_dapo