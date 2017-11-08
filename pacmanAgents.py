# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
from heuristics import scoreEvaluation
import random

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        possible = state.getAllPossibleActions();
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        
        tempState = state;
        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valide actions
        return self.actionList[0];


class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class GreedyAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(scoreEvaluation(state), action) for state, action in successors]
        # get best choice
        bestScore = max(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)

class BFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write BFS Algorithm instead of returning Directions.STOP
        legal = state.getLegalPacmanActions()
        action_list = [(state.generatePacmanSuccessor(action), action) for action in legal]        
        stack = action_list[:]
        while stack:
            curr_state, root_action = stack[0]
            del stack[0]
            legal = curr_state.getLegalPacmanActions()
            successors = [(curr_state.generatePacmanSuccessor(action), root_action) for action in legal]
            for successor in successors:
                if successor[0] is not None:
                    if not curr_state.isWin():
                        stack.append(successor)
                    else:
                        return successor[1]
                else:
                    stack.sort(key = lambda x: scoreEvaluation(x[0]), reverse=False)
                    return stack[-1][1]
        return random.choice(legal)
        

class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write DFS Algorithm instead of returning Directions.STOP
        legal = state.getLegalPacmanActions()
        action_list = [(state.generatePacmanSuccessor(action), action) for action in legal]
        stack = set()
        for action in action_list:
            if action[0].isWin():
                return action[1]
            res = self.DFS(action[0], action[1], stack)
            if res == "WinStateFound":
                return action[1]
            elif res == "IterationLimit":
                best_score = [-pow(2, 32) + 1, Directions.STOP]
                for i in stack:
                    if scoreEvaluation(i[0]) > best_score[0]:
                        best_score = [scoreEvaluation(i[0]), i[1]]
                return best_score[1]
        return random.choice(legal)

    def DFS(self, state, root_action, stack):
        legal = state.getLegalPacmanActions()
        successors = [state.generatePacmanSuccessor(action) for action in legal]
        for i in successors:
            stack.add((i, root_action))
        for successor in successors:
            stack.remove((successor, root_action))
            if successor is None:
                return "IterationLimit"
            if successor.isWin():
                return "WinStateFound"
            res = self.DFS(successor, root_action, stack)
            if res == "WinStateFound" or res == "IterationLimit":
                return res
            

            

class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write A* Algorithm instead of returning Directions.STOP
        legal = state.getLegalPacmanActions()
        action_list = [(state.generatePacmanSuccessor(action), 1, action) for action in legal]
        for action in action_list:
            if action[0].isWin():
                return action[2] 
        root_score = scoreEvaluation(state)
        stack = action_list[:]
        while stack:
            stack.sort(key = lambda x: x[1] - (scoreEvaluation(x[0]) - root_score), reverse=True)
            curr_state, curr_depth, root_action = stack.pop()
            legal = curr_state.getLegalPacmanActions()
            successors = [(curr_state.generatePacmanSuccessor(action), curr_depth + 1, root_action) for action in legal]
            for successor in successors:
                if successor[0] is None:
                    stack.sort(key = lambda x: x[1] - (scoreEvaluation(x[0]) - root_score), reverse=False)
                    return stack[-1][2]
                else:
                    if successor[0].isWin():
                        return successor[2]
                    else:
                        stack.append(successor)
        return random.choice(legal)

class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = []
        self.possible = state.getAllPossibleActions()
        for i in range(0,5):
            self.actionList.append(self.possible[random.randint(0, len(self.possible) - 1)])
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        best_score = -pow(2,32) + 1
        best_play = Directions.STOP
        while True:
            tempState = state
            for action in self.actionList:
                if tempState.isWin() + tempState.isLose() == 0:
                    tempState = tempState.generatePacmanSuccessor(action)
                    if tempState is None:
                        return best_play
                else:
                    if tempState.isWin():
                        return self.actionList[0]
                    break
            temp_score = scoreEvaluation(tempState)
            if temp_score > best_score:
                best_score = temp_score
                best_play = self.actionList[0]
            self.get_next_aciton_sequence()
    
    def get_next_aciton_sequence(self):
        for i in range(0, len(self.actionList) - 1):
            if random.randint(0,1) == 1:
                self.actionList[i] = self.possible[random.randint(0,len(self.possible) - 1)]

class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = []
        self.possible = state.getAllPossibleActions()
        for i in range(0,8):
            self.actionList.append(self.get_random_action_sequece())
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        best_score = -pow(2,32) + 1
        best_play = Directions.STOP
        while True:
            score_list = []
            for i in self.actionList:
                temp_score = self.get_sequence_score(state, i)
                if temp_score == "Over":
                    return best_play
                score_list.append((i, temp_score))
            score_list.sort(key=lambda x:x[1])
            if score_list[-1][1] > best_score:
                best_score = score_list[-1][1]
                best_play = score_list[-1][0][0]
            choose_list = []
            new_sequence = []
            for i in range(1, 9):
                choose_list += [self.actionList[i-1]]*i
            for i in range(4):
                X = random.choice(choose_list)
                Y = random.choice(choose_list)
                if random.randint(1, 100) <= 70:
                    X, Y = self.crossover(X, Y)
                new_sequence.append(X)
                new_sequence.append(Y)
            for i in range(8):
                if random.randint(1, 100) <= 10:
                    self.mutate(new_sequence[i])
            self.actionList = new_sequence[:]


    def mutate(self, chromosomes):
        chromosomes[random.randint(0, len(chromosomes) - 1)] = self.possible[random.randint(0, len(self.possible) - 1)]


    def crossover(self, X, Y):
        result = [[],[]]
        for i in range(2):
            for j in range(5):
                if random.randint(0,1) == 0:
                    result[i].append(X[j])
                else:
                    result[i].append(Y[j])
        return result 


    def get_sequence_score(self, state, actionList):
        tempState = state
        for action in actionList:
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(action)
                if tempState is None:
                    return "Over"
            else:
                break
        return scoreEvaluation(tempState)
            
        
    def get_random_action_sequece(self):
        temp_sequence = []
        for i in range(5):
            temp_sequence.append(self.possible[random.randint(0, len(self.possible) - 1)])
        return temp_sequence


class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        self.root = tree_node(None, None, state.getLegalPacmanActions())
        while True:
            selected_node = self.tree_policy(self.root, state)
            if selected_node =="Over":
                print "#################The MCTS Tree######################"
                self.root.print_tree()
                return self.best_action()
            reward = self.roll_out(state, selected_node)
            if reward == "Over":
                print "#################The MCTS Tree######################"
                self.root.print_tree()
                return self.best_action()
            self.back_propagation(selected_node, reward)


    def tree_policy(self, root, state):
        temp_node = root
        temp_state = state
        while temp_state.isWin() + temp_state.isLose() == 0:
            if len(temp_node.remaining_legal_actions) != 0:
                return self.expand(temp_node, temp_state)
            else:
                temp_node = self.select(temp_node)
                if temp_node is None:
                    return "Over"
                temp_state = temp_state.generatePacmanSuccessor(temp_node.action)
                if temp_state is None:
                    return "Over"
        return temp_node


    def expand(self, node, state):
        action = node.remaining_legal_actions.pop(random.randint(0, len(node.remaining_legal_actions) - 1))
        state = state.generatePacmanSuccessor(action)
        if state is None:
            return "Over"
        return node.add_child(node, action, state.getLegalPacmanActions())
        


    def select(self, node):
        max_score,max_node = -pow(2,32)+1, None
        for child in node.children:
            if child.get_node_score(node) > max_score:
                max_score = child.get_node_score(node)
                max_node = child
        return max_node


    def roll_out(self, state, node):
        temp_node = node
        action_list = []
        while temp_node.parent:
            action_list.append(temp_node.action)
            temp_node = temp_node.parent
        
        temp_state = state
        temp_node = node
        for action in action_list:
            if temp_state.isWin() + temp_state.isLose() == 0:
                temp_state = temp_state.generatePacmanSuccessor(action)
            else:break
            if temp_state is None:
                return "Over"

        for i in range(5):
            if temp_state.isWin() + temp_state.isLose() == 0:
                action_list = temp_state.getLegalPacmanActions()
                action = action_list[random.randint(0, len(action_list) - 1)]
                temp_state = temp_state.generatePacmanSuccessor(action)
                if temp_state is None:
                    return "Over"
            else:break
        return normalizedScoreEvaluation(state, temp_state)


    def back_propagation(self, node, reward):
        temp_node = node
        while temp_node:
            temp_node.reward += reward
            temp_node.visited_num += 1
            temp_node = temp_node.parent

    
    def best_action(self):
        max_visited_num = 0
        action = Directions.STOP
        for child in self.root.children:
            if child.visited_num > max_visited_num:
                max_visited_num = child.visited_num
                action = child.action
        return action


class tree_node(object):
    def __init__(self, parent, up_action, down_legal_actions=[]):
        self.parent = parent
        self.action = up_action
        self.reward = 0.0
        self.visited_num = 0
        self.remaining_legal_actions = down_legal_actions
        self.children = []
    
    def add_child(self, parent, up_action, down_legal_actions=[]):
        child = tree_node(parent, up_action, down_legal_actions)
        self.children.append(child)
        return child

    def get_node_score(self, parent_node, cp=0.05):
        if self.visited_num == 0:
            visited_num = 1
        else:
            visited_num = self.visited_num
        return (self.reward/visited_num) + cp*(2*math.log(parent_node.visited_num)/visited_num)**0.5

    def print_tree(self):
        stack1 = [[self]]
        stack2 = []
        while stack1:
            res = []
            for nodes in stack1:
                for node in nodes:
                    res.append((node.action,node.reward,node.visited_num))
                    stack2.append(node.children)
            print res
            stack1 = stack2
            stack2 = []