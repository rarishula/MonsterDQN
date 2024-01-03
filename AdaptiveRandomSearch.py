import random

def adaptive_random_search(initial_range, steps, shrink_factor, expand_factor):
    # �p�����[�^�͈̔͂�������
    damage_reward_range, front_monster_advantage_reward_range = initial_range

    best_win_rate = 0
    best_params = None

    for _ in range(steps):
        # �p�����[�^�������_���ɑI��
        damage_reward = random.uniform(*damage_reward_range)
        front_monster_advantage_reward = random.uniform(*front_monster_advantage_reward_range)

        # main�֐������s���A�������擾
        win_rate = main(damage_reward, front_monster_advantage_reward)

        # �����V�������������ǂ���΁A�͈͂��X�V
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_params = (damage_reward, front_monster_advantage_reward)

            # �͈͂��k��
            damage_reward_range = shrink_range(damage_reward_range, damage_reward, shrink_factor)
            front_monster_advantage_reward_range = shrink_range(front_monster_advantage_reward_range, front_monster_advantage_reward, shrink_factor)
        else:
            # �͈͂��g��
            damage_reward_range = expand_range(damage_reward_range, expand_factor)
            front_monster_advantage_reward_range = expand_range(front_monster_advantage_reward_range, expand_factor)

    return best_win_rate, best_params

def shrink_range(current_range, current_value, factor):
    """ �͈͂��k�� """
    return (max(current_range[0], current_value - factor * (current_value - current_range[0])),
            min(current_range[1], current_value + factor * (current_range[1] - current_value)))

def expand_range(current_range, factor):
    """ �͈͂��g�� """
    return (current_range[0] - factor * (current_range[1] - current_range[0]),
            current_range[1] + factor * (current_range[1] - current_range[0]))

# �����͈̔͐ݒ�
initial_damage_reward_range = (0, 100)
initial_front_monster_advantage_reward_range = (0, 50)

# �A�_�v�e�B�u�����_���T�[�`�̎��s
best_win_rate, best_params = adaptive_random_search((initial_damage_reward_range, initial_front_monster_advantage_reward_range), 
                                                    steps=100, 
                                                    shrink_factor=0.5, 
                                                    expand_factor=1.2)

print(f"Best Win Rate: {best_win_rate}, Best Params: {best_params}")
