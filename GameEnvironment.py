class GameEnvironment(gym.Env):
    def __init__(self):
        # �v���C���[��AI�̏��������X�^�[��ݒ�
        self.initial_player_monsters = [("Grass", 6), ("Fire", 6), ("Water", 6)]
        self.initial_ai_monsters = [("Grass", 6), ("Fire", 6), ("Water", 6)]

        # ��ԃX�y�[�X�̃T�C�Y��ݒ�i�v���C���[��AI�̃����X�^�[�����킹�����j
        self.state_size = len(self.initial_player_monsters) * 2  # �e�����X�^�[��2�̑����i�^�C�v��HP�j������

        # �A�N�V�����X�y�[�X��ݒ�i��ւƍU���̑I�����j
        self.action_space = gym.spaces.Discrete(4)  # 4�̍s���i2��ނ̍U����2��ނ̌�ցj   def random_action(self, player_monsters):
        valid_actions = get_valid_actions(player_monsters)

        # �e�s���𓯗l�Ɋm���炵���Ƃ���
        action_probability = 1.0 / len(valid_actions) if valid_actions else 0
        return [(action, action_probability) for action in valid_actions]
        
    def reset(self):
        # �����X�^�[�̏�����Ԃ��R�s�[
        self.player_monsters = deepcopy(self.initial_player_monsters)
        self.ai_monsters = deepcopy(self.initial_ai_monsters)
    
        # state���Čv�Z
        self.state = self._convert_to_state(self.player_monsters, self.ai_monsters)
    
        return self.state
        
    def render(self, mode='human'):
        # ���݂̏�Ԃ��e�L�X�g�ŕ\������
        print("���݂̏��:")
        print("�v���C���[�̃����X�^�[:", self.player_monsters)
        print("AI�̃����X�^�[:", self.ai_monsters)

    

        
    def _convert_to_state(self, player_monsters, ai_monsters):
        state = []
        for monster in player_monsters + ai_monsters:
            # �����X�^�[�̓�����ǉ�
            state.append(self._convert_monster_to_features(monster))
        return state
    
    def _convert_monster_to_features(self, monster):
        # �����X�^�[�̃^�C�v�𐔒l�ɕϊ�
        type_to_number = {"Grass": 0, "Fire": 1, "Water": 2}
        type_num = type_to_number[monster[0]]
    
        # �����X�^�[�̗̑�
        hp = monster[1]
    
        return [type_num, hp]
                
    def calculate_next_states_and_probabilities(self, ai_action):
        player_monsters, ai_monsters = deepcopy(self.player_monsters , self.ai_monsters)

        # �v���C���[�̍��@��ƑI���m�����擾
        player_actions_with_select_probs = self.random_action(player_monsters)
        
        # ���@���AI�̍s������s���g�ݍ��킹�ƑI���m�����쐬
        action_combinations_with_select_probs = [(player_action, ai_action, select_prob) for player_action, select_prob in player_actions_with_select_probs]

        next_states_and_probs = []
        for player_action, ai_action, select_prob in action_combinations_with_select_probs:
            action_order = determine_action_order(player_action, ai_action)

            # �U������ȉ��̏ꍇ�̏���
            if sum(action in ["special_attack", "normal_attack"] for _, action in action_order) <= 1:
                temp_player_monsters, temp_ai_monsters = self.apply_actions(player_monsters, ai_monsters, action_order)
                next_states_and_probs.append(((temp_player_monsters, temp_ai_monsters), select_prob))

            # �U���������ꍇ�̏���
            elif sum(action in ["special_attack", "normal_attack"] for _, action in action_order) == 2:
                # �V�i���I1: �v���C���[��U
                temp_player_monsters, temp_ai_monsters = self.apply_actions(player_monsters, ai_monsters, action_order)
                next_states_and_probs.append(((temp_player_monsters, temp_ai_monsters), select_prob * 0.5))

                # �V�i���I2: AI��U
                player_monsters, ai_monsters = deepcopy(state)  # ��Ԃ����Z�b�g
                action_order.reverse()
                temp_player_monsters, temp_ai_monsters = self.apply_actions(player_monsters, ai_monsters, action_order)
                next_states_and_probs.append(((temp_player_monsters, temp_ai_monsters), select_prob * 0.5))

        return next_states_and_probs
        
    def apply_actions(self, player_monsters, ai_monsters, action_order):
        # �^����ꂽ�s�������ɏ]���čs����K�p
        temp_player_monsters, temp_ai_monsters = deepcopy(player_monsters), deepcopy(ai_monsters)
        for side, action in action_order:
            if action.startswith("switch"):
                process_switch(side, action, temp_player_monsters, temp_ai_monsters)
            elif action in ["special_attack", "normal_attack"]:
                if side == "player":
                    temp_player_monsters, temp_ai_monsters, _ = calculate_and_apply_damage(temp_player_monsters, temp_ai_monsters, action)
                else:
                    temp_ai_monsters, temp_player_monsters, _ = calculate_and_apply_damage(temp_ai_monsters, temp_player_monsters, action)
        return temp_player_monsters, temp_ai_monsters
        


    def select_randomly_based_on_probability(next_states_and_probs):
        total_prob = sum(prob for _, prob in next_states_and_probs)
        rand_prob = random.uniform(0, total_prob)
        cumulative_prob = 0
    
        for next_state, prob in next_states_and_probs:
            cumulative_prob += prob
            if cumulative_prob >= rand_prob:
                return next_state
    
        return next_states_and_probs[-1][0]  # ������̏ꍇ�A�Ō�̗v�f��Ԃ�
        
    def calculate_reward(self, next_state):
        # �萔�̐ݒ�
        WIN_REWARD = 100
        LOSE_REWARD = -100
        DAMAGE_REWARD_FACTOR = 50
    
        player_monsters, ai_monsters = self.player_monsters , self.ai_monsters
        next_player_monsters, next_ai_monsters = next_state
    
        # 1. ���s��V
        if all(hp <= 0 for _, hp in next_ai_monsters):
            return WIN_REWARD  # ����
        elif all(hp <= 0 for _, hp in next_player_monsters):
            return LOSE_REWARD  # �s�k
    
        # 2. �_���[�W��V
        damage_reward = DAMAGE_REWARD_FACTOR * (
            sum(hp for _, hp in player_monsters) - sum(hp for _, hp in next_player_monsters)
        )
        damage_taken_reward = DAMAGE_REWARD_FACTOR * (
            sum(hp for _, hp in ai_monsters) - sum(hp for _, hp in next_ai_monsters)
        )
    
        # 3. �Ζʕ�V
        front_monster_advantage_reward = 0
        if is_advantageous(next_player_monsters[0], next_ai_monsters[0]):
            front_monster_advantage_reward += 10
        elif is_advantageous(next_ai_monsters[0], next_player_monsters[0]):
            front_monster_advantage_reward -= 10
    
        return damage_reward - damage_taken_reward + front_monster_advantage_reward
        
    def step(self, action):
        # ���̏�Ԃƕ�V���v�Z����
        next_states_and_probs = self.calculate_next_states_and_probabilities(action)
        next_state = self.select_randomly_based_on_probability(next_states_and_probs)
        reward = self.calculate_reward(next_state)

        return next_state, reward
        
    def is_done(next_state):
        # ���̏�Ԃ̃����X�^�[�̏�Ԃ��擾
        next_player_monsters, next_ai_monsters = next_state
    
        # �v���C���[�̃����X�^�[���S�ē|���ꂽ���ǂ���
        player_all_fainted = all(hp <= 0 for _, hp in next_player_monsters)
    
        # AI�̃����X�^�[���S�ē|���ꂽ���ǂ���
        ai_all_fainted = all(hp <= 0 for _, hp in next_ai_monsters)
    
        # �ǂ��炩���S�ē|���ꂽ�ꍇ�A�Q�[���I��
        return player_all_fainted or ai_all_fainted
        
def is_advantageous(monster1, monster2):
    # �����X�^�[�Ԃ̗L���s���𔻒f����֐�
    # monster1��monster2�͂��ꂼ�ꃂ���X�^�[�̃^�C�v��\��������

    advantage_dict = {
        "Grass": "Water",
        "Water": "Fire",
        "Fire": "Grass"
    }

    if advantage_dict[monster1[0]] == monster2[0]:
        return True  # monster1��monster2�ɑ΂��ėL��
    else:
        return False  # monster1��monster2�ɑ΂��ĕs��