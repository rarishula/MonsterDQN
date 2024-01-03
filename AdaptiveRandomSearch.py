import random

def adaptive_random_search(initial_range, steps, shrink_factor, expand_factor):
    # パラメータの範囲を初期化
    damage_reward_range, front_monster_advantage_reward_range = initial_range

    best_win_rate = 0
    best_params = None

    for _ in range(steps):
        # パラメータをランダムに選択
        damage_reward = random.uniform(*damage_reward_range)
        front_monster_advantage_reward = random.uniform(*front_monster_advantage_reward_range)

        # main関数を実行し、勝率を取得
        win_rate = main(damage_reward, front_monster_advantage_reward)

        # もし新しい勝率がより良ければ、範囲を更新
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_params = (damage_reward, front_monster_advantage_reward)

            # 範囲を縮小
            damage_reward_range = shrink_range(damage_reward_range, damage_reward, shrink_factor)
            front_monster_advantage_reward_range = shrink_range(front_monster_advantage_reward_range, front_monster_advantage_reward, shrink_factor)
        else:
            # 範囲を拡大
            damage_reward_range = expand_range(damage_reward_range, expand_factor)
            front_monster_advantage_reward_range = expand_range(front_monster_advantage_reward_range, expand_factor)

    return best_win_rate, best_params

def shrink_range(current_range, current_value, factor):
    """ 範囲を縮小 """
    return (max(current_range[0], current_value - factor * (current_value - current_range[0])),
            min(current_range[1], current_value + factor * (current_range[1] - current_value)))

def expand_range(current_range, factor):
    """ 範囲を拡大 """
    return (current_range[0] - factor * (current_range[1] - current_range[0]),
            current_range[1] + factor * (current_range[1] - current_range[0]))

# 初期の範囲設定
initial_damage_reward_range = (0, 100)
initial_front_monster_advantage_reward_range = (0, 50)

# アダプティブランダムサーチの実行
best_win_rate, best_params = adaptive_random_search((initial_damage_reward_range, initial_front_monster_advantage_reward_range), 
                                                    steps=100, 
                                                    shrink_factor=0.5, 
                                                    expand_factor=1.2)

print(f"Best Win Rate: {best_win_rate}, Best Params: {best_params}")
