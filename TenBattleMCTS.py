# ���܂ō쐬�����Q�[���̃R�[�h�S�̂�\��
import random
from copy import deepcopy
import math
import pdb  # Python Debugger ���C���|�[�g



# �_���[�W�v�Z�ƓK�p�֐�
def calculate_and_apply_damage(attacker_monsters, defender_monsters, action):
    # �����X�^�[���X�g�̃f�B�[�v�R�s�[���쐬
    attacker_monsters_copy = deepcopy(attacker_monsters)
    defender_monsters_copy = deepcopy(defender_monsters)

    attacker_type = attacker_monsters_copy[0][0]
    defender_type = defender_monsters_copy[0][0]

    # �_���[�W�̌v�Z
    damage = 0
    if action == "special_attack":
        if (attacker_type == "Grass" and defender_type == "Water") or \
           (attacker_type == "Water" and defender_type == "Fire") or \
           (attacker_type == "Fire" and defender_type == "Grass"):
            damage = 6
        else:
            damage = 1
    elif action == "normal_attack":
        damage = 2

    # �_���[�W�̓K�p
    defender_monsters_copy[0] = (defender_monsters_copy[0][0], max(defender_monsters_copy[0][1] - damage, 0))

    # �ύX���ꂽ�R�s�[��Ԃ�
    return attacker_monsters_copy, defender_monsters_copy, damage



# ��֏����֐�
def process_switch(side, action, player_monsters, ai_monsters):
    if "switch" in action:
        switch_to = action.split("_")[1]
        if side == "player":
            new_monster_index = next(i for i, (monster_type, _) in enumerate(player_monsters) if monster_type == switch_to)
            player_monsters[0], player_monsters[new_monster_index] = player_monsters[new_monster_index], player_monsters[0]
        else:
            new_monster_index = next(i for i, (monster_type, _) in enumerate(ai_monsters) if monster_type == switch_to)
            ai_monsters[0], ai_monsters[new_monster_index] = ai_monsters[new_monster_index], ai_monsters[0]





# �^�[���I���֐�
def end_of_turn(player_monsters, ai_monsters, turn_count):
    turn_count += 1

    # �̗͂�0�ȉ��̏ꍇ�̌�㏈��
    for side, monsters in [("player", player_monsters), ("ai", ai_monsters)]:
        if monsters[0][1] <= 0:
            for i, (_, hp) in enumerate(monsters):
                if hp > 0:
                    monsters[0], monsters[i] = monsters[i], monsters[0]
                    break

    # �^�[���I�����̃����X�^�[�̗̑͏�Ԃ�\��
    #print(f"End of Turn {turn_count} State:")
    #for monster in player_monsters:
    #    print(f"Player {monster[0]}: HP {monster[1]}")
    #for monster in ai_monsters:
    #    print(f"AI {monster[0]}: HP {monster[1]}")

    # ���s����
    player_all_fainted = all(hp <= 0 for _, hp in player_monsters)
    ai_all_fainted = all(hp <= 0 for _, hp in ai_monsters)
    if player_all_fainted and ai_all_fainted:
        return "draw", turn_count
    elif player_all_fainted:
        return "ai", turn_count
    elif ai_all_fainted:
        return "player", turn_count
    elif turn_count >= 20:
        return "draw", turn_count

    return None, turn_count  # �Q�[�������s����ꍇ

# �s���I���֐�
def select_random_action(monsters):
    valid_actions = get_valid_actions(monsters)
    return random.choice(valid_actions)


# �s���񋓊֐�
def get_valid_actions(monsters):
    valid_actions = ["special_attack", "normal_attack"]

    # �e�����X�^�[�ɂ��Č�ւ��\���ǂ������`�F�b�N
    for i, (monster_type, hp) in enumerate(monsters):
        if i != 0 and hp > 0:
            valid_actions.append(f"switch_{monster_type}")

    return valid_actions



# �s���g�ݍ��킹�񋓊֐�

def get_action_combinations(player_monsters, ai_monsters):
    player_actions = get_valid_actions(player_monsters)
    ai_actions = get_valid_actions(ai_monsters)
    action_combinations = []

    for player_action in player_actions:
        for ai_action in ai_actions:
            action_combinations.append((player_action, ai_action))

    return action_combinations

class RandomGameSession:
    def __init__(self, player_monsters, ai_monsters):
        self.player_monsters = player_monsters
        self.ai_monsters = ai_monsters

    def play_random_turn(self, turn_count):
        player_action = select_random_action(self.player_monsters)
        ai_action = select_random_action(self.ai_monsters)
        action_order = determine_action_order(player_action, ai_action)

        for side, action in action_order:
            if action.startswith("switch"):
                process_switch(side, action, self.player_monsters, self.ai_monsters)
            elif action in ["special_attack", "normal_attack"]:
                if side == "player":
                    self.player_monsters, self.ai_monsters, _ = calculate_and_apply_damage(self.player_monsters, self.ai_monsters, action)
                else:
                    self.ai_monsters, self.player_monsters, _ = calculate_and_apply_damage(self.ai_monsters, self.player_monsters, action)

        result, turn_count = end_of_turn(self.player_monsters, self.ai_monsters, turn_count)
        return result, turn_count

    def run_game(self):
        turn_count = 0
        result = None
        while not result and turn_count < 20:
            result, turn_count = self.play_random_turn(turn_count)
        return result







# �s����������֐�
def determine_action_order(player_action, ai_action):
    # ����P�[�X: �������U���A�N�V�����̏ꍇ�A�����_���ɏ���������
    if player_action in ["special_attack", "normal_attack"] and ai_action in ["special_attack", "normal_attack"]:
        return sorted([("player", player_action), ("ai", ai_action)], key=lambda x: random.random())
    # ����ȊO�̏ꍇ�A'switch'�A�N�V������D��
    else:
        actions = [("player", player_action), ("ai", ai_action)]
        return sorted(actions, key=lambda x: 0 if "switch" in x[1] else 1)


# �V�~�����[�V����









class Node:
    def __init__(self, player_monsters, ai_monsters, parent=None, action=None, depth = 0, selection_probability=1.0, luck_probability=1.0):
        self.player_monsters = player_monsters
        self.ai_monsters = ai_monsters
        self.parent = parent
        self.action = action
        self.depth = depth
        self.selection_probability = selection_probability
        self.luck_probability = luck_probability
        self.children = []
        self.wins = 0
        self.visits = 0

    def __repr__(self):
        child_str = "\n".join([str(child) for child in self.children])
        return f'Node(Player Monsters: {self.player_monsters}, AI Monsters: {self.ai_monsters}, Action: {self.action}, Depth: {self.depth}, Selection Prob: {self.selection_probability}, Luck Prob: {self.luck_probability}, Wins: {self.wins}, Visits: {self.visits})'

    def get_action_combinations(self):
        # �N���X�O�ɒ�`����Ă��� get_action_combinations �֐����Ăяo��
        return get_action_combinations(self.player_monsters, self.ai_monsters)










    def calculate_grouped_uct(self, exploration_weight=math.sqrt(2)):
        groups = self.group_children_by_ai_action()
        group_uct_values = {}

        for ai_action, group in groups.items():
            total_wins = sum(child.wins for child in group)
            total_visits = sum(child.visits for child in group)

            # �O���[�v�S�̂�UCT�l���v�Z
            if total_visits == 0:
                group_uct_values[ai_action] = float('inf')  # ���T���O���[�v�ɍ����D��x
            else:
                win_rate = total_wins / total_visits
                uct_value = win_rate + exploration_weight * math.sqrt(math.log(self.visits) / total_visits)
                group_uct_values[ai_action] = uct_value

        return group_uct_values








    def best_group(self, exploration_weight=math.sqrt(2)):
        group_uct_values = self.calculate_grouped_uct(exploration_weight)
        best_group_action = max(group_uct_values, key=group_uct_values.get)
        return best_group_action

    def select_from_best_group(self, best_group_action):
        best_group = [child for child in self.children if child.action[1] == best_group_action]
        probabilities = [child.selection_probability * child.luck_probability for child in best_group]
        normalized_probabilities = self.normalize_probabilities(probabilities)
        return random.choices(best_group, weights=normalized_probabilities, k=1)[0]


    def best_child(self, exploration_weight=math.sqrt(2)):
        if len(self.children) == 0:
            return None

        best_group_action = self.best_group(exploration_weight)
        return self.select_from_best_group(best_group_action)



    def add_children_nodes(self):
        action_combinations = self.get_action_combinations()

        # ���̃m�[�h�̏�Ԃ��R�s�[
        original_player_monsters, original_ai_monsters = deepcopy(self.player_monsters), deepcopy(self.ai_monsters)

        # �I���m���͓��l�Ɋm���炵���Ƃ���
        if len(action_combinations) != 0:
            selection_probability = 1.0 / len(action_combinations)
        else:
            selection_probability = 0

        for player_action, ai_action in action_combinations:
            action_order = determine_action_order(player_action, ai_action)

            # �U������ȉ��̏ꍇ�A�ʏ�̍s������
            if sum(action in ["special_attack", "normal_attack"] for _, action in action_order) <= 1:
                temp_player_monsters, temp_ai_monsters = deepcopy(original_player_monsters), deepcopy(original_ai_monsters)
                for side, action in action_order:
                    if action.startswith("switch"):
                        process_switch(side, action, temp_player_monsters, temp_ai_monsters)
                    elif action in ["special_attack", "normal_attack"]:
                        if side == "player":
                            temp_player_monsters, temp_ai_monsters, _ = calculate_and_apply_damage(temp_player_monsters, temp_ai_monsters, action)
                        else:
                            temp_ai_monsters, temp_player_monsters, _ = calculate_and_apply_damage(temp_ai_monsters, temp_player_monsters, action)
                new_node = Node(temp_player_monsters, temp_ai_monsters, self, (player_action, ai_action), self.depth + 1, selection_probability, 1.0)
                self.children.append(new_node)

            # �U���������ꍇ�A�����̃V�i���I���쐬
            if sum(action in ["special_attack", "normal_attack"] for _, action in action_order) == 2:
                # �V�i���I1: �v���C���[��U
                temp_player_monsters, temp_ai_monsters = deepcopy(original_player_monsters), deepcopy(original_ai_monsters)
                for side, action in action_order:
                    if action in ["special_attack", "normal_attack"]:
                        if side == "player":
                            temp_player_monsters, temp_ai_monsters, _ = calculate_and_apply_damage(temp_player_monsters, temp_ai_monsters, action)
                        else:
                            temp_ai_monsters, temp_player_monsters, _ = calculate_and_apply_damage(temp_ai_monsters, temp_player_monsters, action)
                new_node = Node(temp_player_monsters, temp_ai_monsters, self, (player_action, ai_action), self.depth + 1, selection_probability, 0.5)
                self.children.append(new_node)

                # �V�i���I2: AI��U
                temp_player_monsters, temp_ai_monsters = deepcopy(original_player_monsters), deepcopy(original_ai_monsters)
                action_order.reverse()  # �s���������t�ɂ���
                for side, action in action_order:
                    if action in ["special_attack", "normal_attack"]:
                        if side == "player":
                            temp_player_monsters, temp_ai_monsters, _ = calculate_and_apply_damage(temp_player_monsters, temp_ai_monsters, action)
                        else:
                            temp_ai_monsters, temp_player_monsters, _ = calculate_and_apply_damage(temp_ai_monsters, temp_player_monsters, action)
                new_node = Node(temp_player_monsters, temp_ai_monsters, self, (player_action, ai_action), self.depth + 1, selection_probability, 0.5)
                self.children.append(new_node)




    def simulate_game(self):
        # �Q�[���V�~�����[�V�����̂��߂Ƀ����X�^�[�̏�Ԃ̃f�B�[�v�R�s�[���쐬
        player_monsters_copy = deepcopy(self.player_monsters)
        ai_monsters_copy = deepcopy(self.ai_monsters)

        # RandomGameSession�ɃR�s�[��n��
        game_session = RandomGameSession(player_monsters_copy, ai_monsters_copy)
        return game_session.run_game()

    def is_fully_expanded(self):
        # �Q�[�����I�����Ă��邩�ǂ����̃`�F�b�N
        if self.game_is_over():
            return True

        # �[��������̐����ɓ��B���Ă��邩�̃`�F�b�N
        if self.depth >= MAX_DEPTH:
            return True

        # �q�m�[�h�����݂��邩�ǂ����̃`�F�b�N
        return len(self.children) > 0

    def game_is_over(self):
        # �v���C���[�̑S�����X�^�[�̗̑͂�0���ǂ������`�F�b�N
        player_defeated = all(hp <= 0 for _, hp in self.player_monsters)

        # AI�̑S�����X�^�[�̗̑͂�0���ǂ������`�F�b�N
        ai_defeated = all(hp <= 0 for _, hp in self.ai_monsters)

        # �ǂ��炩�̃v���C���[�̃����X�^�[���S�ē|����Ă�����Q�[���I��
        return player_defeated or ai_defeated

    def traverse(self):
    # ���݂̃m�[�h���[���̐����ɒB���Ă��邩�ǂ������`�F�b�N
        if self.depth >= MAX_DEPTH:
            return self

        # ���T���̍s�������邩�ǂ������`�F�b�N
        if not self.is_fully_expanded():
            # ���T���̍s���ɑ΂���V�����q�m�[�h��ǉ�
            self.add_children_nodes()

            # ���T���̎q�m�[�h�������_���ɑI�����ĕԂ�
            return random.choice(self.children)

        next_node = self.best_child()
        if next_node is None:  # 'best_child'��None��Ԃ��ꍇ
            return self  # ����ȏ�T����i�߂Ȃ�
        return next_node.traverse()

    def backpropagate(self, result):
        # ���݂̃m�[�h����e�m�[�h�Ɍ������ăo�b�N�v���p�Q�[�V����
        current_node = self
        while current_node is not None:
            current_node.visits += 1
            if result == "ai":
                current_node.wins += 1
            current_node = current_node.parent

    def group_children_by_ai_action(self):
        # AI�̍s�����ƂɎq�m�[�h���O���[�v��
        groups = {}
        for child in self.children:
            ai_action = child.action[1]  # AI�̍s�����擾
            if ai_action not in groups:
                groups[ai_action] = []
            groups[ai_action].append(child)
        return groups

    def normalize_probabilities(self, probabilities):
        total = sum(probabilities)
        return [p / total if total > 0 else 0 for p in probabilities]


class FullGameSimulation:
    def __init__(self, player_monsters, ai_monsters):
        self.player_monsters = player_monsters
        self.ai_monsters = ai_monsters
        self.turn_count = 0

    def play_turn(self):
        # AI�̍œK�ȍs�������肷��
        root_node = Node(self.player_monsters, self.ai_monsters)
        best_child = MCTS(root_node)
        optimal_ai_action = best_child.action[1]

        # �v���C���[�̍s���������_���ɑI��
        player_action = select_random_action(self.player_monsters)

        # �s���̏���������
        action_order = determine_action_order(player_action, optimal_ai_action)

        # �e�s����K�p
        for side, action in action_order:
            if action.startswith("switch"):
                process_switch(side, action, self.player_monsters, self.ai_monsters)
            elif action in ["special_attack", "normal_attack"]:
                if side == "player":
                    self.player_monsters, self.ai_monsters, _ = calculate_and_apply_damage(self.player_monsters, self.ai_monsters, action)
                else:
                    self.ai_monsters, self.player_monsters, _ = calculate_and_apply_damage(self.ai_monsters, self.player_monsters, action)

        # �^�[���I���̏���
        self.turn_count += 1
        return end_of_turn(self.player_monsters, self.ai_monsters, self.turn_count)

    def run_game(self):
        result = None
        while not result and self.turn_count < 20:
            result = self.play_turn()[0]
            print(f'Player Monsters: {self.player_monsters}, AI Monsters: {self.ai_monsters}\n')

        return result


def MCTS(root):
    NUMBER_OF_ITERATIONS = 1000
    for _ in range(NUMBER_OF_ITERATIONS):
        node = root.traverse()  # �g���o�[�X���Ė��T���̃m�[�h��������
        result = node.simulate_game()  # �V�~�����[�V���������s
        node.backpropagate(result)  # ���ʂ����Ƀo�b�N�v���p�Q�[�V����

    return root.best_child()  # �œK�Ȏq�m�[�h��I��






MAX_DEPTH = 10
ai_win = 0

for i in range(10):
    initial_player_monsters = [("Grass", 6), ("Fire", 6), ("Water", 6)]
    initial_ai_monsters = [("Grass", 6), ("Fire", 6), ("Water", 6)]
    simulation = FullGameSimulation(initial_player_monsters, initial_ai_monsters)
    result = simulation.run_game()
    print(f'Game {i+1}: {result}')
    if result == 'ai':
      ai_win += 1

print(f'ai win :{ai_win}')


#debug_depth_one_nodes(initial_node)
#print(initial_node.children)