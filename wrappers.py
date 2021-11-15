import gym
import numpy as np


def compute_scores(percents):
    # Geometric mean with an offset of 1%.
    scores = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1
    return scores


class CrafterStats(gym.Wrapper):

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        info['episode_extra_stats'] = info.get('episode_extra_stats', {})
        if done:
            achievements = []
            for achievement in info['achievements']:
                achievements.append(100.0 if info['achievements'][achievement] > 0.0 else 0.0)
                info['episode_extra_stats'][achievement] = achievements[-1]

            info['episode_extra_stats']['Score'] = compute_scores(np.array(achievements))
            info['episode_extra_stats']['Num_achievements'] = int(sum(achievements) // 100.0)
        return obs, reward, done, info
