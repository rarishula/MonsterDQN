import gym
from gym.spaces import MultiDiscrete
import random
from copy import deepcopy
import math
import numpy as np

class GameEnvironment(gym.Env):
    #def __init__(self,damage_reward_range, front_monster_advantage_reward_range):
    def __init__(self):
        # プレイヤーとAIの初期モンスターを設定
        self.initial_player_monsters = [("Grass", 6), ("Fire", 6), ("Water", 6)]
        self.initial_ai_monsters = [("Grass", 6), ("Fire", 6), ("Water", 6)]

        # 状態スペースのサイズを設定（プレイヤーとAIのモンスターを合わせた数）
        self.state_size = len(self.initial_player_monsters) * 2  # 各モンスターは2つの属性（タイプとHP）を持つ
        self.observation_space = MultiDiscrete([3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7])# 例: モンスターの種類が 3 種類、体力が 0 から 6 の範囲

        # アクションスペースを設定（交替と攻撃の選択肢）
        self.action_space = gym.spaces.Discrete(4)  # 4つの行動（2種類の攻撃と2種類の交替）

        self.player_wins = 0
        self.ai_wins = 0
        self.draws = 0
        #self.damage_reward_range = damage_reward_range
        #self.front_monster_advantage_reward_range = front_monster_advantage_reward_range


        
    def random_action(self, player_monsters):
        valid_actions = get_valid_actions(player_monsters)

        # 各行動を同様に確からしいとする
        action_probability = 1.0 / len(valid_actions) if valid_actions else 0
        return [(action, action_probability) for action in valid_actions]
        
    def reset(self):
        # モンスターの初期状態をコピー
        self.player_monsters = deepcopy(self.initial_player_monsters)
        self.ai_monsters = deepcopy(self.initial_ai_monsters)
    
        # stateを再計算（平坦化して1次元の長さ12の配列にする）
        self.state = self._convert_to_state(self.player_monsters, self.ai_monsters)
        flat_state = [feature for monster in self.state for feature in monster]
    
        return flat_state
        
    def render(self, mode='human'):
        # 現在の状態をテキストで表示する
        #print("Current State:")
        #print("Player Monster:", self.player_monsters)
        #print("Ai Monster:", self.ai_monsters)
        
        # 勝敗が決まった場合の表示
        result = self.check_game_result()
        if result:
            print(f"Result: {result}")
            print("Player Wins:", self.player_wins, "AI Wins:", self.ai_wins, "Draws:", self.draws)
        
    def _convert_to_state(self, player_monsters, ai_monsters):
        state = []
        for monster in player_monsters + ai_monsters:
            # モンスターの特徴を追加
            state.append(self._convert_monster_to_features(monster))
        return state
    
    def _convert_monster_to_features(self, monster):
        # モンスターのタイプを数値に変換
        type_to_number = {"Grass": 0, "Fire": 1, "Water": 2}
        type_num = type_to_number[monster[0]]
    
        # モンスターの体力
        hp = monster[1]
    
        return [type_num, hp]
                
    def calculate_next_states_and_probabilities(self, ai_action):
        player_monsters = deepcopy(self.player_monsters)
        ai_monsters = deepcopy(self.ai_monsters)


        # プレイヤーの合法手と選択確率を取得
        player_actions_with_select_probs = self.random_action(player_monsters)
        
        # 合法手とAIの行動から行動組み合わせと選択確率を作成
        action_combinations_with_select_probs = [(player_action, ai_action, select_prob) for player_action, select_prob in player_actions_with_select_probs]

        next_states_and_probs = []
        for player_action, ai_action, select_prob in action_combinations_with_select_probs:
            action_order = determine_action_order(player_action, ai_action)

            # 攻撃が一つ以下の場合の処理
            if sum(action in ["special_attack", "normal_attack"] for _, action in action_order) <= 1:
                temp_player_monsters, temp_ai_monsters = self.apply_actions(player_monsters, ai_monsters, action_order)
                next_states_and_probs.append(((temp_player_monsters, temp_ai_monsters), select_prob))

            # 攻撃が二つある場合の処理
            elif sum(action in ["special_attack", "normal_attack"] for _, action in action_order) == 2:
                # シナリオ1: プレイヤー先攻
                temp_player_monsters, temp_ai_monsters = self.apply_actions(player_monsters, ai_monsters, action_order)
                next_states_and_probs.append(((temp_player_monsters, temp_ai_monsters), select_prob * 0.5))

                # シナリオ2: AI先攻
                player_monsters = deepcopy(self.player_monsters)
                ai_monsters = deepcopy(self.ai_monsters)
                action_order.reverse()
                temp_player_monsters, temp_ai_monsters = self.apply_actions(player_monsters, ai_monsters, action_order)
                next_states_and_probs.append(((temp_player_monsters, temp_ai_monsters), select_prob * 0.5))

        return next_states_and_probs
        
    def apply_actions(self, player_monsters, ai_monsters, action_order):
        # 与えられた行動順序に従って行動を適用
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
        

    @staticmethod
    def select_randomly_based_on_probability(next_states_and_probs):
        total_prob = sum(prob for _, prob in next_states_and_probs)
        rand_prob = random.uniform(0, total_prob)
        cumulative_prob = 0
    
        for next_state, prob in next_states_and_probs:
            cumulative_prob += prob
            if cumulative_prob >= rand_prob:
                return next_state
    
        return next_states_and_probs[-1][0]  # 万が一の場合、最後の要素を返す
        
    def calculate_reward(self, next_state):
        # 定数の設定
        WIN_REWARD = 100
        LOSE_REWARD = -100
        #DAMAGE_REWARD_FACTOR = self.damage_reward_range
        DAMAGE_REWARD_FACTOR = 50
        player_monsters, ai_monsters = self.player_monsters , self.ai_monsters
        next_player_monsters, next_ai_monsters = next_state
    
        # 1. 勝敗報酬
        if all(hp <= 0 for _, hp in next_ai_monsters):
            return WIN_REWARD  # 勝利
        elif all(hp <= 0 for _, hp in next_player_monsters):
            return LOSE_REWARD  # 敗北
    
        # 2. ダメージ報酬
        damage_reward = DAMAGE_REWARD_FACTOR * (
            sum(hp for _, hp in player_monsters) - sum(hp for _, hp in next_player_monsters)
        )
        damage_taken_reward = DAMAGE_REWARD_FACTOR * (
            sum(hp for _, hp in ai_monsters) - sum(hp for _, hp in next_ai_monsters)
        )
    
        # 3. 対面報酬
        front_monster_advantage_reward = 0
        if is_advantageous(next_player_monsters[0], next_ai_monsters[0]):
            #front_monster_advantage_reward += self.front_monster_advantage_reward_range
            front_monster_advantage_reward += 10
        elif is_advantageous(next_ai_monsters[0], next_player_monsters[0]):
            #front_monster_advantage_reward -= self.front_monster_advantage_reward_range
            front_monster_advantage_reward -= 10
    
        return damage_reward - damage_taken_reward + front_monster_advantage_reward
        
    def step(self, action):
        # 整数のactionを文字列に変換
        if action == 0:
            ai_action = "special_attack"
        elif action == 1:
            ai_action = "normal_attack"
        elif action == 2:
            ai_action = "switch_1"
        elif action == 3:
            ai_action = "switch_2"

        # 次の状態と報酬を計算する
        next_states_and_probs = self.calculate_next_states_and_probabilities(ai_action)
        next_state = self.select_randomly_based_on_probability(next_states_and_probs)

        player_monsters, ai_monsters = next_state
        done = end_of_turn(player_monsters, ai_monsters)
        next_state = (player_monsters, ai_monsters)

        reward = self.calculate_reward(next_state)


        player_monsters, ai_monsters = next_state
        
        # next_stateを1次元の配列に平坦化する
        converted_next_state = self._convert_to_state(player_monsters, ai_monsters)

        flat_next_state = [feature for monster in converted_next_state for feature in monster]
        self.state = flat_next_state
        self.player_monsters, self.ai_monsters = next_state


        # 追加情報（空の辞書）
        info = {}

        if done == "player":
            self.player_wins += 1
        elif done == "ai":
            self.ai_wins += 1
        elif done == "draw":
            self.draws += 1

        return self.state, reward, done, info
        
    def reset_win_counts(self):
        # 勝敗数をリセットする
        self.player_wins = 0
        self.ai_wins = 0
        self.draws = 0

    def check_game_result(self):
        player_all_fainted = all(hp <= 0 for _, hp in self.player_monsters)
        ai_all_fainted = all(hp <= 0 for _, hp in self.ai_monsters)

        if player_all_fainted and ai_all_fainted:
            return "draw"
        elif player_all_fainted:
            return "ai"
        elif ai_all_fainted:
            return "player"
        return None  # ゲームが続行する場合



        
def is_advantageous(monster1, monster2):
    # モンスター間の有利不利を判断する関数
    # monster1とmonster2はそれぞれモンスターのタイプを表す文字列

    advantage_dict = {
        "Grass": "Water",
        "Water": "Fire",
        "Fire": "Grass"
    }

    if advantage_dict[monster1[0]] == monster2[0]:
        return True  # monster1はmonster2に対して有利
    else:
        return False  # monster1はmonster2に対して不利

def get_valid_actions(monsters):
    valid_actions = ["special_attack", "normal_attack"]

    # 各モンスターについて交替が可能かどうかをチェック
    for i, (_, hp) in enumerate(monsters):
        if i != 0 and hp > 0:
            valid_actions.append(f"switch_{i}")

    return valid_actions

    
def determine_action_order(player_action, ai_action):
    # 特殊ケース: 両方が攻撃アクションの場合、ランダムに順序を決定
    if player_action in ["special_attack", "normal_attack"] and ai_action in ["special_attack", "normal_attack"]:
        return sorted([("player", player_action), ("ai", ai_action)], key=lambda x: random.random())
    # それ以外の場合、'switch'アクションを優先
    else:
        actions = [("player", player_action), ("ai", ai_action)]
        return sorted(actions, key=lambda x: 0 if "switch" in x[1] else 1)

def process_switch(side, action, player_monsters, ai_monsters):
    if "switch" in action:
        # actionから交替対象のモンスターのインデックスを取得
        switch_to_index = int(action.split("_")[1])

        if side == "player":
            # プレイヤー側のモンスターを交替
            player_monsters[0], player_monsters[switch_to_index] = player_monsters[switch_to_index], player_monsters[0]
        else:
            # AI側のモンスターを交替
            ai_monsters[0], ai_monsters[switch_to_index] = ai_monsters[switch_to_index], ai_monsters[0]


def calculate_and_apply_damage(attacker_monsters, defender_monsters, action):
    # モンスターリストのディープコピーを作成
    attacker_monsters_copy = deepcopy(attacker_monsters)
    defender_monsters_copy = deepcopy(defender_monsters)

    attacker_type = attacker_monsters_copy[0][0]
    attacker_hp = attacker_monsters_copy[0][1]
    defender_type = defender_monsters_copy[0][0]

    # 攻撃側モンスターの体力が0の場合、ダメージは0
    if attacker_hp <= 0:
        return attacker_monsters_copy, defender_monsters_copy, 0

    # ダメージの計算
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

    # ダメージの適用
    defender_monsters_copy[0] = (defender_monsters_copy[0][0], max(defender_monsters_copy[0][1] - damage, 0))

    # 変更されたコピーを返す
    return attacker_monsters_copy, defender_monsters_copy, damage
    
def end_of_turn(player_monsters, ai_monsters):

    # 体力が0以下の場合の交代処理
    for side, monsters in [("player", player_monsters), ("ai", ai_monsters)]:
        if monsters[0][1] <= 0:
            for i, (_, hp) in enumerate(monsters):
                if hp > 0:
                    monsters[0], monsters[i] = monsters[i], monsters[0]
                    break


    # 勝敗判定
    player_all_fainted = all(hp <= 0 for _, hp in player_monsters)
    ai_all_fainted = all(hp <= 0 for _, hp in ai_monsters)
    if player_all_fainted and ai_all_fainted:
        return "draw"
    elif player_all_fainted:
        return "ai"
    elif ai_all_fainted:
        return "player"


    return None  # ゲームが続行する場合
